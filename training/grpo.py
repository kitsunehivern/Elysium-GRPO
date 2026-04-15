import copy
import json
import math
import types
import os
import os.path as osp
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import torch.nn.functional as F
import transformers
from easydict import EasyDict as edict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import nullcontext

from data.processors.box_processor import BOX_PROCESSORS
from data.video_llm_data import VideoLLMPredictProcessor
from reward.grpo_reward import (
    parse_box_from_raw_text,
    compute_step_rewards,
    extract_step_end_token_indices,
    build_process_advantages,
    compute_sequence_reward,
)


@dataclass
class ModelArguments:
    model: Optional[dict] = field(default_factory=dict)


@dataclass
class DataArguments:
    data: Optional[dict] = field(default_factory=dict)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    beta: float = field(default=0.1)
    remove_unused_columns: bool = field(default=False)
    visual_encoder_lr_scale: float = field(default=1.0)


class GRPOClipDataset(Dataset):
    """
    Same idea as your LocalDataset, but for GRPO round 1 we ALWAYS use the GT init box
    and never use global_box_pool.
    """
    BOX_STR_TEMPLATE = "Frame {i}: <box>"
    FRAME_STR_TEMPLATE = "Frame {i}: <image>"

    SOT_QUESTION_TEMPLATE = (
        "{frame_str}This is a video showing an object with coordinates <box> in Frame 1. "
        "Please provide the detailed coordinates of the object in each frame."
    )
    RSOT_QUESTION_TEMPLATE = (
        "{frame_str}Please find one {object_class} and provide the detailed coordinates in each frame."
    )

    def __init__(self, image_folder, anno_path, clip_len=8, task="SOT", processor=None):
        self.image_folder = image_folder
        self.processor = processor
        self.clip_len = clip_len
        self.task = task
        self.box_processor = BOX_PROCESSORS["ours_v1"]

        with open(anno_path, "r", encoding="utf-8") as f:
            anns = [json.loads(line) for line in f]
        self.preprocess(anns)

    def preprocess(self, anns):
        self.anns = []
        for item in anns:
            frames_path = item["frames"]
            boxes = item["box"]
            assert len(boxes) == len(frames_path)

            n_clip = math.ceil(len(frames_path) / (self.clip_len - 1))
            for clip_id in range(n_clip):
                clip_frames_path = frames_path[
                    clip_id * (self.clip_len - 1): clip_id * (self.clip_len - 1) + self.clip_len
                ]
                clip_boxes = boxes[
                    clip_id * (self.clip_len - 1): clip_id * (self.clip_len - 1) + self.clip_len
                ]

                # keep only full-enough clips
                if len(clip_frames_path) < 2:
                    continue

                clip_data = dict(
                    frames_path=clip_frames_path,
                    box=clip_boxes,
                    initial_box=clip_boxes[0],
                    frame_size=item["frame_size"],
                    object_description=item.get("object_description", ""),
                    object_class=item.get("object_class", "object"),
                    seq_id=item["vid"],
                    clip_id=clip_id,
                )
                self.anns.append(clip_data)

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        item = copy.deepcopy(self.anns[idx])
        initial_box = item["initial_box"]

        frame_len = len(item["frames_path"])
        frame_str = ", ".join(self.FRAME_STR_TEMPLATE.format(i=i + 1) for i in range(frame_len)) + "\n"
        box_str = ", ".join(self.BOX_STR_TEMPLATE.format(i=i + 1) for i in range(frame_len))

        if self.task == "SOT":
            question = self.SOT_QUESTION_TEMPLATE.format(frame_str=frame_str)
            answer = box_str
        elif self.task == "RSOT":
            question = self.RSOT_QUESTION_TEMPLATE.format(frame_str=frame_str, object_class=item["object_class"])
            answer = box_str
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        if self.box_processor.box_token in question:
            if question.count(self.box_processor.box_token) == 1:
                question = self.box_processor(question, [initial_box])
            else:
                question = self.box_processor(question, item["box"])

        if self.box_processor.box_token in answer:
            if answer.count(self.box_processor.box_token) == 1:
                answer = self.box_processor(answer, [item["initial_box"]])
            else:
                answer = self.box_processor(answer, item["box"])

        messages = [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ]

        data_dict = {
            "id": f"{item['seq_id']}|{item['clip_id']}",
            "vid": f"{item['seq_id']}|{item['clip_id']}",
            "frames": item["frames_path"],
            "image_folder": self.image_folder,
            "question": question,
            "gt": answer,
            "vqa": messages,
            "image_size": item["frame_size"],
        }

        output = self.processor.transform(data_dict)
        output["gt_boxes_raw"] = item["box"]
        output["raw_question"] = question
        output["raw_gt"] = answer
        output["raw_image_size"] = item["frame_size"]
        return output

def patch_llm_forward_ignore_bad_labels(model):
    """
    Elysium's custom forward sometimes passes a Python list into self.llm(labels=...).
    For GRPO we do not want CE loss anyway, only logits, so force labels=None unless
    labels is already a real tensor.
    """
    orig_forward = model.llm.forward

    def safe_forward(self, *args, **kwargs):
        labels = kwargs.get("labels", None)
        if labels is not None and not torch.is_tensor(labels):
            # print(f"[patch_llm_forward_ignore_bad_labels] forcing labels=None from type={type(labels)}")
            kwargs["labels"] = None
        return orig_forward(*args, **kwargs)

    model.llm.forward = types.MethodType(safe_forward, model.llm)
    return model

def maybe_autocast(device: str, compute_dtype):
    if device.startswith("cuda") and compute_dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=compute_dtype)
    return nullcontext()

def get_tokenizer_path(model_args, data_args):
    if isinstance(model_args.model, dict) and model_args.model.get("tokenizer_name_or_path", None):
        return model_args.model["tokenizer_name_or_path"]
    dp = edict(data_args.data).predict.data_preprocess
    if hasattr(dp, "tokenizer"):
        return dp.tokenizer
    return model_args.model["model_name_or_path"]

def resolve_compute_dtype(training_args, device: str):
    if device.startswith("cuda"):
        if getattr(training_args, "bf16", False):
            return torch.bfloat16
        if getattr(training_args, "fp16", False):
            return torch.float16
        return torch.float32
    return torch.float32

def forward_for_logits(model, frames, n_frames, input_ids, attention_mask):
    """
    Use forward() only for logits, never for LM loss.
    Force labels=None so Elysium does not enter the LLaMA loss path.
    """
    if isinstance(input_ids, torch.Tensor):
        assert input_ids.ndim == 2, f"input_ids should be [B, L], got {tuple(input_ids.shape)}"

    if isinstance(frames, torch.Tensor):
        assert frames.ndim in (4, 5), f"frames should look like image/video tensors, got {tuple(frames.shape)}"

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        frames=frames,
        n_frames=n_frames,
        labels=None,          # <-- critical fix
        return_dict=True,
        use_cache=False,
    )

    if isinstance(outputs, tuple):
        logits = outputs[0]
    elif hasattr(outputs, "logits"):
        logits = outputs.logits
    else:
        raise ValueError(f"Cannot extract logits from model output of type {type(outputs)}")

    return logits


def repeat_visual(value, times):
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return value.repeat(times)
        return value.repeat(times, *([1] * (value.dim() - 1)))
    elif isinstance(value, list):
        return value * times
    else:
        return value


def masked_mean(x, mask):
    denom = mask.float().sum().clamp_min(1.0)
    return (x * mask.float()).sum() / denom


def shift_logprobs_from_logits(logits, input_ids):
    """
    Compute token logprobs for next-token prediction, while safely handling
    negative / out-of-range token ids (e.g. multimodal sentinel ids like -200).

    Returns:
        tok_logps: [B, L-1]
        valid_next_mask: [B, L-1] boolean mask for gather-valid positions
    """
    assert logits.ndim == 3, f"logits must be [B, L, V], got {tuple(logits.shape)}"
    assert input_ids.ndim == 2, f"input_ids must be [B, L], got {tuple(input_ids.shape)}"
    assert logits.size(0) == input_ids.size(0), (
        f"Batch mismatch: logits {tuple(logits.shape)} vs input_ids {tuple(input_ids.shape)}"
    )

    seq_len = min(logits.size(1), input_ids.size(1))
    logits = logits[:, :seq_len, :]
    input_ids = input_ids[:, :seq_len]

    if seq_len < 2:
        raise ValueError(f"Sequence too short for shifted logprobs: seq_len={seq_len}")

    vocab_size = logits.size(-1)
    next_tokens = input_ids[:, 1:]                  # [B, L-1]

    valid_next_mask = (next_tokens >= 0) & (next_tokens < vocab_size)

    # replace invalid ids with a safe dummy index so gather does not crash
    safe_next_tokens = next_tokens.masked_fill(~valid_next_mask, 0).unsqueeze(-1)

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    tok_logps = torch.gather(log_probs, dim=-1, index=safe_next_tokens).squeeze(-1)

    # zero out invalid positions explicitly
    tok_logps = tok_logps.masked_fill(~valid_next_mask, 0.0)

    return tok_logps, valid_next_mask


def build_full_sequences_from_text(
    batch,
    sample_idx,
    texts,
    tokenizer,
    device,
    max_total_len=768,
    append_eos=True,
):
    prompt_ids = batch["input_ids"][sample_idx][batch["attention_mask"][sample_idx].bool()].tolist()

    all_ids = []
    all_completion_masks = []
    completion_token_lens = []

    for text in texts:
        completion_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if append_eos and tokenizer.eos_token_id is not None:
            completion_ids = completion_ids + [tokenizer.eos_token_id]

        full_ids = prompt_ids + completion_ids
        full_ids = full_ids[:max_total_len]

        prompt_len = min(len(prompt_ids), len(full_ids))
        comp_len = max(0, len(full_ids) - prompt_len)

        completion_mask = [0] * prompt_len + [1] * comp_len

        all_ids.append(torch.tensor(full_ids, dtype=torch.long))
        all_completion_masks.append(torch.tensor(completion_mask, dtype=torch.long))
        completion_token_lens.append(comp_len)

    max_len = max(x.size(0) for x in all_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    input_ids = torch.full((len(all_ids), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(all_ids), max_len), dtype=torch.long, device=device)
    completion_mask = torch.zeros((len(all_ids), max_len), dtype=torch.long, device=device)

    for i, (ids, cmask) in enumerate(zip(all_ids, all_completion_masks)):
        L = ids.size(0)
        input_ids[i, :L] = ids.to(device)
        attention_mask[i, :L] = 1
        completion_mask[i, :L] = cmask.to(device)

    return input_ids, attention_mask, completion_mask, completion_token_lens


def sample_group_texts(
    model,
    single_batch,
    group_size,
    temperature,
    top_p,
    max_new_tokens,
):
    n_frames = single_batch["n_frames"]
    if torch.is_tensor(n_frames):
        n_frames_list = n_frames.detach().cpu().tolist()
    else:
        n_frames_list = list(n_frames)

    if isinstance(single_batch["frames"], torch.Tensor):
        num_frame_tensors = single_batch["frames"].shape[0]
        assert sum(n_frames_list) == num_frame_tensors, (
            f"Mismatch between frames and n_frames: "
            f"sum(n_frames)={sum(n_frames_list)} vs frames.shape[0]={num_frame_tensors}"
        )

    generate_params = dict(
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        min_length=4,
        repetition_penalty=1.0,
        length_penalty=1.0,
    )

    # First try num_return_sequences
    try:
        outputs = model.generate(
            single_batch["frames"],
            single_batch["n_frames"],
            single_batch["input_ids"],
            single_batch["attention_mask"],
            num_return_sequences=group_size,
            **generate_params,
        )
        if isinstance(outputs, list) and len(outputs) == group_size:
            return outputs
        if isinstance(outputs, torch.Tensor):
            raise ValueError("Custom generate returned tensor, expected decoded strings.")
    except Exception:
        pass

    # Fallback: loop group_size times
    texts = []
    for _ in range(group_size):
        out = model.generate(
            single_batch["frames"],
            single_batch["n_frames"],
            single_batch["input_ids"],
            single_batch["attention_mask"],
            **generate_params,
        )
        if isinstance(out, list):
            texts.append(out[0])
        else:
            texts.append(out)
    return texts

def sample_group_completions(
    model,
    single_batch,
    group_size,
    temperature,
    top_p,
    max_new_tokens,
):
    assert single_batch["input_ids"].size(0) == 1, "Expected batch size 1 for GRPO."

    vision_encode_out = model._encode_vision(single_batch["frames"], single_batch["n_frames"])
    inputs_embeds, mm_attention_mask, _ = model._concat_embedding(
        vision_encode_out,
        single_batch["input_ids"],
        single_batch["attention_mask"],
        labels=None,
    )

    pad_id = model.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = model.tokenizer.eos_token_id

    texts = []
    completion_tensors = []
    completion_lens = []

    for _ in range(group_size):
        gen_out = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=mm_attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_token_id=model.tokenizer.eos_token_id,
            pad_token_id=pad_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        gen_len = len(gen_out.scores)
        if gen_len == 0:
            comp = torch.empty((0,), dtype=torch.long, device=single_batch["input_ids"].device)
            text = ""
        else:
            comp = gen_out.sequences[0, -gen_len:]
            text = model.tokenizer.decode(comp, skip_special_tokens=True).strip()

        completion_tensors.append(comp)
        completion_lens.append(int(comp.numel()))
        texts.append(text)

    max_len = max(completion_lens) if completion_lens else 0
    completion_ids = torch.full(
        (group_size, max_len),
        pad_id,
        dtype=torch.long,
        device=single_batch["input_ids"].device,
    )

    for i, comp in enumerate(completion_tensors):
        if comp.numel() > 0:
            completion_ids[i, : comp.numel()] = comp

    return texts, completion_ids, completion_lens


def build_full_sequences_from_completion_ids(
    batch,
    sample_idx,
    completion_ids,
    completion_lens,
    tokenizer,
    device,
    max_total_len=768,
):
    prompt_ids = batch["input_ids"][sample_idx][batch["attention_mask"][sample_idx].bool()]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    all_ids = []
    all_completion_masks = []
    used_completion_lens = []

    for i in range(completion_ids.size(0)):
        comp = completion_ids[i, : completion_lens[i]]
        full_ids = torch.cat([prompt_ids, comp], dim=0)[:max_total_len]

        prompt_len = min(prompt_ids.numel(), full_ids.numel())
        comp_len = max(0, full_ids.numel() - prompt_len)

        cmask = torch.zeros(full_ids.size(0), dtype=torch.long, device=device)
        if comp_len > 0:
            cmask[prompt_len : prompt_len + comp_len] = 1

        all_ids.append(full_ids)
        all_completion_masks.append(cmask)
        used_completion_lens.append(comp_len)

    max_len = max(x.size(0) for x in all_ids)

    input_ids = torch.full((len(all_ids), max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(all_ids), max_len), dtype=torch.long, device=device)
    completion_mask = torch.zeros((len(all_ids), max_len), dtype=torch.long, device=device)

    for i, (ids, cmask) in enumerate(zip(all_ids, all_completion_masks)):
        L = ids.size(0)
        input_ids[i, :L] = ids
        attention_mask[i, :L] = 1
        completion_mask[i, :L] = cmask

    return input_ids, attention_mask, completion_mask, used_completion_lens


def prepare_multimodal_inputs(
    model,
    frames,
    n_frames,
    input_ids,
    attention_mask,
    completion_mask,
):
    vision_encode_out = model._encode_vision(frames, n_frames)

    inputs_embeds, mm_attention_mask, targets = model._concat_embedding(
        vision_encode_out,
        input_ids,
        attention_mask,
        labels=input_ids,
    )

    _, _, completion_targets = model._concat_embedding(
        vision_encode_out,
        input_ids,
        attention_mask,
        labels=completion_mask.long(),
    )

    return inputs_embeds, mm_attention_mask, targets, completion_targets


def shift_logprobs_from_logits_and_targets(logits, targets):
    assert logits.ndim == 3, f"logits must be [B, L, V], got {tuple(logits.shape)}"
    assert targets.ndim == 2, f"targets must be [B, L], got {tuple(targets.shape)}"

    seq_len = min(logits.size(1), targets.size(1))
    logits = logits[:, :seq_len, :]
    targets = targets[:, :seq_len]

    next_tokens = targets[:, 1:]
    valid_next_mask = (next_tokens >= 0) & (next_tokens < logits.size(-1))

    safe_next_tokens = next_tokens.masked_fill(~valid_next_mask, 0).unsqueeze(-1)
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    tok_logps = torch.gather(log_probs, dim=-1, index=safe_next_tokens).squeeze(-1)
    tok_logps = tok_logps.masked_fill(~valid_next_mask, 0.0)

    return tok_logps, valid_next_mask

class GRPOTrainerMinimal:
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        dataloader,
        optimizer,
        task,
        group_size=4,
        temperature=0.8,
        top_p=0.95,
        clip_eps=0.2,
        kl_beta=0.04,
        max_new_tokens=256,
        save_dir="./outputs/grpo",
        save_steps=200,
        device="cuda",
        compute_dtype=torch.float32,
    ):
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.task = task
        self.group_size = group_size
        self.temperature = temperature
        self.top_p = top_p
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta
        self.max_new_tokens = max_new_tokens
        self.save_dir = save_dir
        self.save_steps = save_steps
        self.device = device
        self.compute_dtype = compute_dtype

        self.model.eval()
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def save_checkpoint(self, step):
        save_path = osp.join(self.save_dir, f"checkpoint-step-{step}")
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def train(self, max_steps=1000):
        os.makedirs(self.save_dir, exist_ok=True)
        global_step = 0

        progress = tqdm(total=max_steps, desc="GRPO")
        while global_step < max_steps:
            for batch in self.dataloader:
                if global_step >= max_steps:
                    break

                assert len(batch["id"]) == 1, "Set predict/data_fetch/batch_sizes: [1] for first GRPO run."

                # print("frames dtype =", batch["frames"].dtype, "shape =", tuple(batch["frames"].shape))
                # print("input_ids dtype =", batch["input_ids"].dtype, "shape =", tuple(batch["input_ids"].shape))

                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        if k == "frames":
                            batch[k] = v.to(self.device, dtype=self.compute_dtype)
                        else:
                            batch[k] = v.to(self.device)

                # IMPORTANT:
                # Do NOT slice frames/n_frames like ordinary batch-major tensors.
                # In this repo, frames are packed multimodal inputs whose length must match n_frames.
                single = batch

                # with torch.no_grad():
                #     texts = sample_group_texts(
                #         self.model.eval(),
                #         single_batch=single,
                #         group_size=self.group_size,
                #         temperature=self.temperature,
                #         top_p=self.top_p,
                #         max_new_tokens=self.max_new_tokens,
                #     )
                #     # self.model.train()

                # image_size = [float(v) for v in batch["raw_image_size"][0]]
                # gt_boxes_abs = batch["gt_boxes_raw"][0]

                # input_ids, attention_mask, completion_mask, completion_token_lens = build_full_sequences_from_text(
                #     batch=batch,
                #     sample_idx=0,
                #     texts=texts,
                #     tokenizer=self.tokenizer,
                #     device=self.device,
                #     max_total_len=768,
                # )

                # # Repeat the full visual input bundle for G sampled candidates
                # frames_rep = repeat_visual(batch["frames"], self.group_size)
                # n_frames_rep = repeat_visual(batch["n_frames"], self.group_size)

                # with torch.no_grad():
                #     if isinstance(frames_rep, torch.Tensor):
                #         nf = n_frames_rep.detach().cpu().tolist() if torch.is_tensor(n_frames_rep) else list(n_frames_rep)
                #         assert sum(nf) == frames_rep.shape[0], (
                #             f"Repeated visual bundle mismatch: sum(n_frames_rep)={sum(nf)} "
                #             f"vs frames_rep.shape[0]={frames_rep.shape[0]}"
                #         )

                #     old_logits = forward_for_logits(
                #         self.model.eval(),
                #         frames_rep,
                #         n_frames_rep,
                #         input_ids,
                #         attention_mask,
                #     )
                #     ref_logits = forward_for_logits(
                #         self.ref_model,
                #         frames_rep,
                #         n_frames_rep,
                #         input_ids,
                #         attention_mask,
                #     )
                #     # self.model.train()

                # old_logps, valid_next_mask = shift_logprobs_from_logits(old_logits, input_ids)
                # if global_step == 0:
                #     invalid_ids = input_ids[:, 1:][~valid_next_mask]
                #     if invalid_ids.numel() > 0:
                #         print("DEBUG invalid next-token ids:", torch.unique(invalid_ids).detach().cpu().tolist())
                #         print("DEBUG invalid count:", int((~valid_next_mask).sum().item()))
                # ref_logps, _ = shift_logprobs_from_logits(ref_logits, input_ids)

                # step_rewards_group = []
                # step_end_indices_group = []
                # diagnostics = []

                # for text in texts:
                #     if global_step == 0:
                #         print("DEBUG gt_boxes_raw type:", type(gt_boxes_abs))
                #         if torch.is_tensor(gt_boxes_abs):
                #             print("DEBUG gt_boxes_raw tensor shape:", tuple(gt_boxes_abs.shape))
                #         else:
                #             print("DEBUG gt_boxes_raw preview:", str(gt_boxes_abs)[:300])

                #     pred_boxes = parse_box_from_raw_text(text)
                #     reward_info = compute_step_rewards(pred_boxes, gt_boxes_abs, image_size)
                #     step_rewards_group.append(reward_info["step_rewards"])
                #     diagnostics.append(reward_info)
                #     step_end_indices_group.append(extract_step_end_token_indices(text, self.tokenizer))

                # # align number of step boundaries with GT frames
                # T = len(gt_boxes_abs)
                # step_rewards_group = [x[:T] + ([x[-1]] * max(0, T - len(x)) if len(x) > 0 else [0.0] * T) for x in step_rewards_group]
                # step_end_indices_group = [
                #     x[:T] + ([x[-1]] * max(0, T - len(x)) if len(x) > 0 else [completion_token_lens[i]] * T)
                #     for i, x in enumerate(step_end_indices_group)
                # ]

                # advantages_list = build_process_advantages(
                #     completion_token_lens=completion_token_lens,
                #     step_end_token_indices=step_end_indices_group,
                #     step_rewards=step_rewards_group,
                # )

                # # pack advantages to padded tensor matching input_ids
                # advantages_full = torch.zeros_like(input_ids, dtype=torch.float)
                # for i, adv in enumerate(advantages_list):
                #     prompt_len = int((completion_mask[i] == 0).sum().item())
                #     adv = adv[:completion_token_lens[i]]
                #     if len(adv) > 0:
                #         advantages_full[i, prompt_len:prompt_len + len(adv)] = torch.tensor(adv, dtype=torch.float, device=self.device)

                # # current policy logprobs
                # cur_logits = forward_for_logits(
                #     self.model,
                #     frames_rep,
                #     n_frames_rep,
                #     input_ids,
                #     attention_mask,
                # )
                # cur_logps, _ = shift_logprobs_from_logits(cur_logits, input_ids)

                # # shift masks to token-prediction positions
                # completion_mask_shift = completion_mask[:, 1:].float() * valid_next_mask.float()
                # advantages_shift = advantages_full[:, 1:]

                # ratio = torch.exp(cur_logps - old_logps)
                # surr1 = ratio * advantages_shift
                # surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_shift
                # policy_loss = -masked_mean(torch.min(surr1, surr2), completion_mask_shift)

                # # DeepSeekMath-style direct KL estimator:
                # # exp(log_pref - log_pcur) - (log_pref - log_pcur) - 1 :contentReference[oaicite:9]{index=9}
                # log_ratio_ref = ref_logps - cur_logps
                # kl_term = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
                # kl_loss = masked_mean(kl_term, completion_mask_shift)

                # loss = policy_loss + self.kl_beta * kl_loss

                with torch.no_grad():
                    texts, completion_ids, completion_lens = sample_group_completions(
                        self.model,
                        single_batch=single,
                        group_size=self.group_size,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_new_tokens=self.max_new_tokens,
                    )

                image_size = [float(v) for v in batch["raw_image_size"][0]]
                gt_boxes = batch["gt_boxes_raw"][0]

                input_ids, attention_mask, completion_mask, completion_token_lens = (
                    build_full_sequences_from_completion_ids(
                        batch=batch,
                        sample_idx=0,
                        completion_ids=completion_ids,
                        completion_lens=completion_lens,
                        tokenizer=self.tokenizer,
                        device=self.device,
                        max_total_len=768,
                    )
                )

                frames_rep = repeat_visual(batch["frames"], self.group_size)
                n_frames_rep = repeat_visual(batch["n_frames"], self.group_size)

                with torch.no_grad():
                    old_inputs_embeds, old_mm_attn, old_targets, old_completion_targets = prepare_multimodal_inputs(
                        self.model,
                        frames_rep,
                        n_frames_rep,
                        input_ids,
                        attention_mask,
                        completion_mask,
                    )
                    old_logits = self.model.llm(
                        inputs_embeds=old_inputs_embeds,
                        attention_mask=old_mm_attn,
                        return_dict=True,
                        use_cache=False,
                    ).logits

                    ref_inputs_embeds, ref_mm_attn, ref_targets, _ = prepare_multimodal_inputs(
                        self.ref_model,
                        frames_rep,
                        n_frames_rep,
                        input_ids,
                        attention_mask,
                        completion_mask,
                    )
                    ref_logits = self.ref_model.llm(
                        inputs_embeds=ref_inputs_embeds,
                        attention_mask=ref_mm_attn,
                        return_dict=True,
                        use_cache=False,
                    ).logits

                old_logps, valid_next_mask = shift_logprobs_from_logits_and_targets(old_logits, old_targets)
                ref_logps, _ = shift_logprobs_from_logits_and_targets(ref_logits, ref_targets)

                diagnostics = []
                scalar_rewards = []
                ignore_first_frame = (self.task == "SOT")

                for text in texts:
                    pred_boxes = parse_box_from_raw_text(text)
                    reward_info = compute_sequence_reward(
                        pred_boxes_norm=pred_boxes,
                        gt_boxes=gt_boxes,
                        image_size=image_size,
                        ignore_first_frame=ignore_first_frame,
                    )
                    diagnostics.append(reward_info)
                    scalar_rewards.append(reward_info["sequence_reward"])

                rewards = torch.tensor(scalar_rewards, dtype=torch.float32, device=self.device)
                advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)
                advantages_shift = advantages.unsqueeze(1).expand_as(old_logps)

                cur_inputs_embeds, cur_mm_attn, cur_targets, cur_completion_targets = prepare_multimodal_inputs(
                    self.model,
                    frames_rep,
                    n_frames_rep,
                    input_ids,
                    attention_mask,
                    completion_mask,
                )
                cur_logits = self.model.llm(
                    inputs_embeds=cur_inputs_embeds,
                    attention_mask=cur_mm_attn,
                    return_dict=True,
                    use_cache=False,
                ).logits
                cur_logps, _ = shift_logprobs_from_logits_and_targets(cur_logits, cur_targets)

                completion_mask_shift = (cur_completion_targets[:, 1:] > 0).float() * valid_next_mask.float()

                ratio = torch.exp(cur_logps - old_logps)
                surr1 = ratio * advantages_shift
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_shift
                policy_loss = -masked_mean(torch.min(surr1, surr2), completion_mask_shift)

                log_ratio_ref = ref_logps - cur_logps
                kl_term = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
                kl_term = torch.clamp(kl_term, min=-100.0, max=100.0)
                kl_loss = masked_mean(kl_term, completion_mask_shift)

                loss = policy_loss + self.kl_beta * kl_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                if global_step == 0:
                    print("==================================== DEBUG INFO FOR FIRST STEP ====================================")
                    print("num gt boxes =", len(gt_boxes))
                    print("raw image size =", image_size)
                    print("completion_ids.shape =", tuple(completion_ids.shape))
                    print("unexpanded input_ids.shape =", tuple(input_ids.shape))
                    print("expanded target shape =", tuple(cur_targets.shape))
                    print("active completion tokens =", int(completion_mask_shift.sum().item()))
                    print("sample rewards =", scalar_rewards)
                    print("====================================================================================================")
                else:
                    print(f"Step {global_step}: reward mean={rewards.mean().item():.4f} std={rewards.std().item():.4f} "
                          f"policy_loss={policy_loss.item():.4f} kl_loss={kl_loss.item():.4f}")

                global_step += 1
                progress.update(1)

                best_idx = max(range(len(diagnostics)), key=lambda i: diagnostics[i]["mean_iou"])
                progress.set_postfix(
                    step=global_step,
                    loss=float(loss.item()),
                    policy=float(policy_loss.item()),
                    kl=float(kl_loss.item()),
                    best_iou=float(diagnostics[best_idx]["mean_iou"]),
                    valid=float(diagnostics[best_idx]["format_valid_rate"]),
                )

                if global_step % self.save_steps == 0:
                    self.save_checkpoint(global_step)

        progress.close()
        self.save_checkpoint(global_step)

def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def freeze_for_grpo(model, lm_lr_scale=1.0):
    set_requires_grad(model.visual_encoder, False)
    set_requires_grad(model.llm, False)

    if hasattr(model, "adapter"):
        set_requires_grad(model.adapter, True)
    else:
        raise AttributeError(
            "Cannot find model.adapter. "
            "Search model.named_parameters() for adapter / selector / projector names."
        )

    return model

def main():
    global local_rank

    argument_parser = ArgumentParser()
    argument_parser.add_argument("--config", type=str, required=True)
    argument_parser.add_argument("--task", type=str, choices=("SOT", "RSOT"), default="SOT")
    argument_parser.add_argument("--group_size", type=int, default=4)
    argument_parser.add_argument("--temperature", type=float, default=0.8)
    argument_parser.add_argument("--top_p", type=float, default=0.95)
    argument_parser.add_argument("--clip_eps", type=float, default=0.2)
    argument_parser.add_argument("--kl_beta", type=float, default=0.04)
    argument_parser.add_argument("--max_new_tokens", type=int, default=256)
    argument_parser.add_argument("--max_steps", type=int, default=1000)
    argument_parser.add_argument("--save_steps", type=int, default=200)
    argument_parser.add_argument("--lr", type=float, default=1e-6)
    argument_parser.add_argument("--save_dir", type=str, default="./outputs/grpo_sot")
    argument_parser.add_argument('--local_rank', type=int)
    args = argument_parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_rank = args.local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_yaml_file(args.config)

    model_path = model_args.model["model_name_or_path"]
    tokenizer_path = get_tokenizer_path(model_args, data_args)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = resolve_compute_dtype(training_args, device)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
    ).to(device)
    
    model = freeze_for_grpo(model)

    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            print("TRAIN:", name, tuple(p.shape))
    print(f"trainable params: {trainable:,} / {total:,} = {100 * trainable / total:.4f}%")

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
    ).to(device)

    if compute_dtype == torch.bfloat16:
        model = model.bfloat16()
        ref_model = ref_model.bfloat16()
    elif compute_dtype == torch.float16:
        model = model.half()
        ref_model = ref_model.half()
    else:
        model = model.float()
        ref_model = ref_model.float()

    model = patch_llm_forward_ignore_bad_labels(model)
    ref_model = patch_llm_forward_ignore_bad_labels(ref_model)

    # optional but useful for consistency/debugging
    model.torch_dtype = compute_dtype
    ref_model.torch_dtype = compute_dtype
    if hasattr(model, "config"):
        model.config.torch_dtype = compute_dtype
    if hasattr(ref_model, "config"):
        ref_model.config.torch_dtype = compute_dtype

    print("compute_dtype =", compute_dtype)
    print("model param dtype =", next(model.parameters()).dtype)
    print("ref param dtype =", next(ref_model.parameters()).dtype)

    dp_config = edict(data_args.data).predict.data_preprocess
    df_config = edict(data_args.data).predict.data_fetch
    extra_meta_keys = [
        "source",
        "id",
        "question",
        "gt",
        "gt_boxes_raw",
        "raw_question",
        "raw_gt",
        "raw_image_size",
    ]
    existing_meta_keys = list(getattr(dp_config, "meta_keys", []))
    dp_config.update({"meta_keys": list(dict.fromkeys(existing_meta_keys + extra_meta_keys))})
    processor = VideoLLMPredictProcessor(**dp_config)

    dataset = GRPOClipDataset(
        image_folder=df_config.image_folder,
        anno_path=df_config.anno_path,
        processor=processor,
        task=args.task,
        clip_len=8,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,   # first-pass GRPO: one prompt => one sampled group
        shuffle=True,
        num_workers=0,
        collate_fn=processor.batch_transform,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    trainer = GRPOTrainerMinimal(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        optimizer=optimizer,
        task=args.task,
        group_size=args.group_size,
        temperature=args.temperature,
        top_p=args.top_p,
        clip_eps=args.clip_eps,
        kl_beta=args.kl_beta,
        max_new_tokens=args.max_new_tokens,
        save_dir=args.save_dir,
        save_steps=args.save_steps,
        device=device,
        compute_dtype=compute_dtype,
    )
    trainer.train(max_steps=args.max_steps)


if __name__ == "__main__":
    main()

"""
smoke:
CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=/raid/hvtham/dcmquan/Elysium \
python training/grpo.py \
  --config /raid/hvtham/dcmquan/Elysium/configs/grpo_sot_smoke.yaml \
  --task SOT \
  --group_size 4 \
  --temperature 0.8 \
  --top_p 0.95 \
  --clip_eps 0.2 \
  --kl_beta 0.04 \
  --lr 2e-3 \
  --max_new_tokens 256 \
  --max_steps 1000 \
  --save_steps 200 \
  --save_dir /raid/hvtham/dcmquan/Elysium/outputs/grpo_sot_smoke

full:
CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=/raid/hvtham/dcmquan/Elysium \
python training/grpo.py \
  --config /raid/hvtham/dcmquan/Elysium/configs/grpo_sot.yaml \
  --task SOT \
  --group_size 4 \
  --temperature 0.8 \
  --top_p 0.95 \
  --clip_eps 0.2 \
  --kl_beta 0.04 \
  --lr 1e-4 \
  --max_new_tokens 256 \
  --max_steps 5000 \
  --save_steps 500 \
  --save_dir /raid/hvtham/dcmquan/Elysium/outputs/grpo_sot_run_1
"""
