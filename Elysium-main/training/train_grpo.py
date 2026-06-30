from __future__ import annotations

import copy
import json
import math
import os
import random
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from easydict import EasyDict as edict
from PIL import Image
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer import is_sagemaker_mp_enabled

from data.video_llm_data import VideoLLMProcessor
from models.modeling_elysium import ElysiumConfig, ElysiumForCausalLM
from training.grpo_rewards import (
    FinalRewardWeights,
    RewardConfig,
    TrackingComponentWeights,
    compute_batch_tracking_rewards,
    extract_video_description_span,
)


@dataclass
class ModelArguments:
    model: Optional[dict] = field(default_factory=dict)


@dataclass
class DataArguments:
    data: Optional[dict] = field(default_factory=dict)


@dataclass
class GRPOArguments:
    grpo: Optional[dict] = field(default_factory=dict)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    visual_encoder_lr_scale: float = field(default=1.0)
    remove_unused_columns: bool = field(default=False)
    using_torch_lr: bool = field(default=False)
    lr_type: str = field(default="")


class LocalDataset(Dataset):
    """Same lightweight JSONL dataset wrapper used by training/train.py."""

    def __init__(self, data_paths, multi_round_qa=True, processor=None):
        self.anns = []
        for data_path in data_paths:
            image_folder = data_path["image_folder"]
            anno_path = data_path["anno_path"]
            with open(anno_path, "r") as f:
                if multi_round_qa:
                    for line in f:
                        item = json.loads(line)
                        item["image_folder"] = image_folder
                        self.anns.append(item)
                else:
                    for line in f:
                        line = json.loads(line)
                        # If the annotation already has no multi-round VQA field,
                        # keep it intact so OnlineVQAProcessor can construct SOT/RSOT.
                        if "vqa" not in line:
                            line["image_folder"] = image_folder
                            self.anns.append(line)
                            continue
                        vqas = line["vqa"]
                        num_rounds = len(vqas) // 2
                        line.pop("vqa")
                        for i in range(num_rounds):
                            single_round_line = copy.deepcopy(line)
                            single_round_line["vqa"] = vqas[2 * i : 2 * i + 2]
                            single_round_line["image_folder"] = image_folder
                            self.anns.append(single_round_line)
        self.processor = processor

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        item = copy.deepcopy(self.anns[idx])
        return self.processor.transform(item)


def unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def split_concatenated_frames(frames: torch.Tensor, n_frames: Sequence[int]) -> List[torch.Tensor]:
    chunks = []
    cursor = 0
    for n in n_frames:
        n = int(n)
        chunks.append(frames[cursor : cursor + n])
        cursor += n
    return chunks


def repeat_visual_batch(frames: torch.Tensor, n_frames: Sequence[int], repeats: int) -> Tuple[torch.Tensor, List[int]]:
    chunks = split_concatenated_frames(frames, n_frames)
    repeated_chunks: List[torch.Tensor] = []
    repeated_n_frames: List[int] = []
    for chunk, n in zip(chunks, n_frames):
        for _ in range(repeats):
            repeated_chunks.append(chunk)
            repeated_n_frames.append(int(n))
    return torch.cat(repeated_chunks, dim=0), repeated_n_frames


def repeat_frame_chunks(frames: torch.Tensor, n_frames: Sequence[int], repeats: int) -> List[torch.Tensor]:
    chunks = split_concatenated_frames(frames, n_frames)
    out = []
    for chunk in chunks:
        for _ in range(repeats):
            out.append(chunk)
    return out


def pad_1d_tensors(tensors: Sequence[torch.Tensor], pad_value: int, device: torch.device) -> torch.Tensor:
    max_len = max(int(t.numel()) for t in tensors)
    out = torch.full((len(tensors), max_len), pad_value, dtype=torch.long, device=device)
    for i, t in enumerate(tensors):
        out[i, : t.numel()] = t.to(device=device, dtype=torch.long)
    return out


def make_attention_mask(input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    return input_ids.ne(pad_id).long()


def trim_generated_tokens(
    tokens: Sequence[int],
    eos_id: Optional[int],
    pad_id: Optional[int],
    bos_id: Optional[int],
) -> List[int]:
    """Keep only newly generated answer tokens.

    When a decoder-only HF model is called with ``inputs_embeds`` but without
    ``input_ids``, ``generate`` may return a dummy BOS token followed by the
    generated continuation.  If the model immediately emits EOS, returning EOS
    as the whole completion decodes to an empty string and later creates a fake
    one-token training target.  In that case we return an empty completion so the
    reward becomes zero and the group can be skipped cleanly.
    """
    toks = [int(t) for t in tokens]
    while toks and pad_id is not None and toks[0] == pad_id:
        toks.pop(0)
    if toks and bos_id is not None and toks[0] == bos_id:
        toks = toks[1:]
    # If generation stopped immediately, this is not a useful completion.
    if len(toks) == 1 and eos_id is not None and toks[0] == eos_id:
        return []
    if eos_id is not None and eos_id in toks:
        toks = toks[: toks.index(eos_id) + 1]
    toks = [t for t in toks if pad_id is None or t != pad_id]
    return toks


class SigLIPSemanticRewarder:
    """Optional VideoRFT-style visual semantic consistency scorer.

    The score mirrors VideoRFT Eq. 5: min(1, w * max(cos(text, video), 0)).
    It is intentionally lazy and disabled by default because loading SigLIP adds
    memory cost.  Use it only when the model is prompted to emit <think> text.
    """

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = edict(cfg or {})
        self.enabled = bool(self.cfg.get("enabled", False))
        self.device = device
        self.model = None
        self.processor = None
        self.mean = torch.tensor(self.cfg.get("image_mean", [0.48145466, 0.4578275, 0.40821073])).view(3, 1, 1)
        self.std = torch.tensor(self.cfg.get("image_std", [0.26862954, 0.26130258, 0.27577711])).view(3, 1, 1)
        self.max_frames = int(self.cfg.get("max_frames", 4))
        self.text_span_words = int(self.cfg.get("text_span_words", 96))
        self.scale = float(self.cfg.get("scale", 2.0))

    def _lazy_load(self):
        if not self.enabled or self.model is not None:
            return
        from transformers import AutoModel, AutoProcessor

        model_name = self.cfg.get("model_name_or_path", "siglip-so400m-patch14-384")
        dtype_name = str(self.cfg.get("torch_dtype", "bf16")).lower()
        dtype = torch.bfloat16 if dtype_name in {"bf16", "bfloat16"} else torch.float16 if dtype_name in {"fp16", "float16"} else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

    def _frames_to_pil(self, frame_chunk: torch.Tensor) -> List[Image.Image]:
        if frame_chunk.numel() == 0:
            return []
        frame_chunk = frame_chunk.detach().float().cpu()
        # Pick evenly spaced frames for efficiency.
        if frame_chunk.shape[0] > self.max_frames:
            idx = torch.linspace(0, frame_chunk.shape[0] - 1, self.max_frames).round().long().tolist()
            frame_chunk = frame_chunk[idx]
        mean = self.mean.to(frame_chunk)
        std = self.std.to(frame_chunk)
        frames = frame_chunk * std + mean
        frames = frames.clamp(0, 1)
        pil_frames = []
        for frame in frames:
            arr = (frame.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
            pil_frames.append(Image.fromarray(arr))
        return pil_frames

    @torch.no_grad()
    def __call__(self, completions: Sequence[str], frame_chunks: Sequence[torch.Tensor]) -> List[float]:
        if not self.enabled:
            return [0.0] * len(completions)
        self._lazy_load()
        assert self.model is not None and self.processor is not None

        rewards: List[float] = []
        for text, frames in zip(completions, frame_chunks):
            desc = extract_video_description_span(text, max_words=self.text_span_words)
            if not desc:
                rewards.append(0.0)
                continue
            pil_frames = self._frames_to_pil(frames)
            if not pil_frames:
                rewards.append(0.0)
                continue
            inputs = self.processor(
                images=pil_frames,
                text=[desc] * len(pil_frames),
                return_tensors="pt",
                truncation=True,
                padding="max_length",
            ).to(self.device)
            outputs = self.model(**inputs)
            image_embeds = F.normalize(outputs.image_embeds.float(), dim=-1)
            text_embeds = F.normalize(outputs.text_embeds.float(), dim=-1)
            sim = (image_embeds * text_embeds).sum(dim=-1).mean()
            reward = torch.minimum(torch.ones_like(sim), self.scale * torch.maximum(sim, torch.zeros_like(sim)))
            rewards.append(float(reward.detach().cpu()))
        return rewards


class TrackGRPOTrainer(Trainer):
    def __init__(self, processor: VideoLLMProcessor, grpo_config: dict, reference_model=None, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
        self.grpo_config = edict(grpo_config or {})
        self.reference_model = reference_model
        if self.reference_model is not None:
            self.reference_model.eval()
            for p in self.reference_model.parameters():
                p.requires_grad = False

        reward_cfg = edict(self.grpo_config.get("reward", {}))
        tracking_w = edict(reward_cfg.get("tracking_weights", {}))
        final_w = edict(reward_cfg.get("final_weights", {}))
        self.reward_cfg = RewardConfig(
            tracking_weights=TrackingComponentWeights(
                iou=float(tracking_w.get("iou", 0.65)),
                center=float(tracking_w.get("center", 0.15)),
                temporal=float(tracking_w.get("temporal", 0.10)),
                validity=float(tracking_w.get("validity", 0.10)),
            ),
            final_weights=FinalRewardWeights(
                format=float(final_w.get("format", 0.10)),
                accuracy=float(final_w.get("accuracy", 0.90)),
                semantic=float(final_w.get("semantic", 0.00)),
            ),
            coordinate_scale=float(reward_cfg.get("coordinate_scale", 100.0)),
            center_tau=float(reward_cfg.get("center_tau", 10.0)),
            temporal_tau=float(reward_cfg.get("temporal_tau", 20.0)),
            count_mismatch_penalty=float(reward_cfg.get("count_mismatch_penalty", 0.50)),
            clamp_for_metrics=bool(reward_cfg.get("clamp_for_metrics", True)),
            semantic_gate=float(reward_cfg.get("semantic_gate", 0.05)),
            format_style=str(reward_cfg.get("format_style", "answer_only")),
            require_frame_prefix=bool(reward_cfg.get("require_frame_prefix", False)),
        )
        self.semantic_rewarder = None

    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            scale_lr_parameters = [
                p for n, p in opt_model.named_parameters() if n.startswith("visual_encoder") and p.requires_grad
            ]
            optimizer_grouped_parameters = []

            decay_params = [
                p
                for n, p in opt_model.named_parameters()
                if n in decay_parameters
                and not n.startswith("visual_encoder")
                and p.requires_grad
            ]

            no_decay_params = [
                p
                for n, p in opt_model.named_parameters()
                if n not in decay_parameters
                and not n.startswith("visual_encoder")
                and p.requires_grad
            ]

            visual_params = [
                p
                for n, p in opt_model.named_parameters()
                if n.startswith("visual_encoder") and p.requires_grad
            ]

            if len(decay_params) > 0:
                optimizer_grouped_parameters.append(
                    {
                        "params": decay_params,
                        "weight_decay": self.args.weight_decay,
                    }
                )

            if len(no_decay_params) > 0:
                optimizer_grouped_parameters.append(
                    {
                        "params": no_decay_params,
                        "weight_decay": 0.0,
                    }
                )

            if len(visual_params) > 0:
                optimizer_grouped_parameters.append(
                    {
                        "params": visual_params,
                        "weight_decay": 0.0,
                        "lr": self.args.visual_encoder_lr_scale * self.args.learning_rate,
                    }
                )

            print("Optimizer parameter groups:", len(optimizer_grouped_parameters))
            for i, group in enumerate(optimizer_grouped_parameters):
                n_params = sum(p.numel() for p in group["params"])
                print(
                    f"  group {i}: params={n_params}, "
                    f"weight_decay={group.get('weight_decay')}, "
                    f"lr={group.get('lr', self.args.learning_rate)}"
                )

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
        return self.optimizer

    def _append_prompt_suffix(self, prompts: Sequence[str]) -> List[str]:
        """Insert extra generation instructions into the user turn.

        Elysium prompts usually end with ``ASSISTANT:``.  Appending a suffix
        after that marker makes the suffix look like assistant-generated text,
        which often teaches the LLM that the answer has already started and can
        make it emit EOS immediately.  Insert the suffix before the final
        assistant marker instead.
        """
        suffix = str(self.grpo_config.get("prompt_suffix", "")).strip()
        if not suffix:
            return list(prompts)

        assistant_markers = ["ASSISTANT:", "ASSISTANT: "]
        fixed: List[str] = []
        for prompt in prompts:
            inserted = False
            for marker in assistant_markers:
                idx = prompt.rfind(marker)
                if idx >= 0:
                    prefix = prompt[:idx].rstrip()
                    fixed.append(prefix + "\n" + suffix + "\n" + prompt[idx:])
                    inserted = True
                    break
            if not inserted:
                fixed.append(prompt.rstrip() + "\n" + suffix)
        return fixed

    def _tokenize_prompts(self, prompts: Sequence[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_id = self.processor.pad_id
        ids = [torch.as_tensor(self.processor.tokenizer_vision_placeholder(p), dtype=torch.long) for p in prompts]
        input_ids = pad_1d_tensors(ids, pad_value=pad_id, device=device)
        attention_mask = make_attention_mask(input_ids, pad_id=pad_id)
        return input_ids, attention_mask

    @torch.no_grad()
    def _generate_completions(self, model, batch: Dict[str, torch.Tensor]) -> Tuple[List[List[int]], List[str]]:
        base_model = unwrap_model(model)
        tokenizer = base_model.tokenizer
        num_generations = int(self.grpo_config.get("num_generations", 4))
        max_new_tokens = int(self.grpo_config.get("max_new_tokens", 256))
        min_new_tokens = int(self.grpo_config.get("min_new_tokens", 4))
        temperature = float(self.grpo_config.get("temperature", 0.7))
        top_p = float(self.grpo_config.get("top_p", 0.9))
        do_sample = bool(self.grpo_config.get("do_sample", True))

        prompts = self._append_prompt_suffix(batch["prompt"])
        expanded_prompts = [p for p in prompts for _ in range(num_generations)]
        frames_rep, n_frames_rep = repeat_visual_batch(batch["frames"], batch["n_frames"], num_generations)
        prompt_ids, prompt_attention = self._tokenize_prompts(expanded_prompts, device=batch["frames"].device)

        vision_encode_out = base_model._encode_vision(frames_rep, n_frames_rep)
        inputs_embeds, attention_mask, _ = base_model._concat_embedding(
            vision_encode_out, prompt_ids, prompt_attention, labels=None
        )

        generate_kwargs = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        # The original Elysium evaluator used a minimum length.  Without this,
        # Vicuna/LLaMA can choose EOS as the first generated token, producing
        # tensors like [BOS, EOS] and an empty decoded completion.
        if min_new_tokens > 0:
            generate_kwargs["min_new_tokens"] = min_new_tokens

        llm_was_training = base_model.llm.training
        base_model.llm.eval()

        try:
            generated = base_model.llm.generate(**generate_kwargs)
        finally:
            base_model.llm.train(llm_was_training)

        completion_token_ids = [
            trim_generated_tokens(
                row.tolist(),
                eos_id=tokenizer.eos_token_id,
                pad_id=tokenizer.pad_token_id,
                bos_id=tokenizer.bos_token_id,
            )
            for row in generated
        ]
        completion_texts = tokenizer.batch_decode(completion_token_ids, skip_special_tokens=True)
        return completion_token_ids, completion_texts

    def _build_full_sequences(
        self,
        prompts: Sequence[str],
        completion_token_ids: Sequence[Sequence[int]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pad_id = self.processor.pad_id
        prompts = self._append_prompt_suffix(prompts)
        full_ids: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        for prompt, completion in zip(prompts, completion_token_ids):
            prompt_ids = self.processor.tokenizer_vision_placeholder(prompt)
            completion = list(completion)
            ids = torch.as_tensor(prompt_ids + completion, dtype=torch.long)
            lab = torch.full_like(ids, fill_value=-100)
            if len(completion) > 0:
                lab[len(prompt_ids) :] = torch.as_tensor(completion, dtype=torch.long)
            full_ids.append(ids)
            labels.append(lab)
        input_ids = pad_1d_tensors(full_ids, pad_value=pad_id, device=device)
        labels = pad_1d_tensors(labels, pad_value=-100, device=device)
        attention_mask = make_attention_mask(input_ids, pad_id=pad_id)
        return input_ids, attention_mask, labels

    def _per_token_logps(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        frames: torch.Tensor,
        n_frames: Sequence[int],
        force_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base_model = unwrap_model(model)

        if force_dtype is not None:
            frames = frames.to(dtype=force_dtype)
        
        vision_encode_out = base_model._encode_vision(frames, n_frames)
        inputs_embeds, attention_mask, targets = base_model._concat_embedding(
            vision_encode_out, input_ids, attention_mask, labels=labels
        )
        outputs = base_model.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False,
        )
        logits = outputs.logits[:, :-1, :].float()
        shifted_targets = targets[:, 1:]
        mask = shifted_targets.ne(-100)
        safe_targets = shifted_targets.masked_fill(~mask, 0)
        logps = F.log_softmax(logits, dim=-1).gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
        return logps, mask.float()

    def _get_semantic_rewarder(self, device: torch.device) -> SigLIPSemanticRewarder:
        if self.semantic_rewarder is None:
            sem_cfg = edict(self.grpo_config.get("semantic_reward", {}))
            self.semantic_rewarder = SigLIPSemanticRewarder(sem_cfg, device=device)
        return self.semantic_rewarder

    def _compute_rewards(self, completion_texts: Sequence[str], batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        num_generations = int(self.grpo_config.get("num_generations", 4))
        gt_expanded = [gt for gt in batch["gt"] for _ in range(num_generations)]

        semantic_rewards = None
        if float(self.reward_cfg.final_weights.semantic) > 0.0:
            frame_chunks = repeat_frame_chunks(batch["frames"], batch["n_frames"], num_generations)
            semantic_rewards = self._get_semantic_rewarder(batch["frames"].device)(completion_texts, frame_chunks)

        rewards, metrics = compute_batch_tracking_rewards(
            completion_texts,
            gt_expanded,
            cfg=self.reward_cfg,
            semantic_rewards=semantic_rewards,
        )
        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=batch["frames"].device)
        return reward_tensor, metrics

    def _group_advantages(self, rewards: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards_g = rewards.view(-1, group_size)
        mean = rewards_g.mean(dim=1, keepdim=True)
        std = rewards_g.std(dim=1, keepdim=True, unbiased=False)
        advantages = (rewards_g - mean) / (std + 1e-6)

        min_std = float(self.grpo_config.get("min_reward_std", 1e-4))
        skip_zero_std = bool(self.grpo_config.get("skip_zero_std_groups", True))
        valid_group = (std.squeeze(1) >= min_std).float()
        if skip_zero_std:
            advantages = advantages * valid_group.view(-1, 1)
        return advantages.view(-1).detach(), valid_group.detach()

    def _get_named_checkpoint_base_path(self) -> str:
        """Base path for thesis-friendly checkpoints, e.g. checkpoints/grpo_uav123."""
        return str(
            self.grpo_config.get("trained_model_name_or_path")
            or self.grpo_config.get("named_checkpoint_base_path")
            or ""
        ).strip()

    def _save_named_model_checkpoint(self, step: int):
        """Save an additional model checkpoint named <base>_<step>.

        This is separate from HuggingFace/DeepSpeed's normal
        output_dir/checkpoint-<step> checkpoints, so `save_total_limit` can delete
        old training-state checkpoints without deleting these thesis/evaluation
        model snapshots.
        """
        base_path = self._get_named_checkpoint_base_path()
        if not base_path or step <= 0:
            return

        save_steps = int(getattr(self.args, "save_steps", 0) or 0)
        if save_steps <= 0 or step % save_steps != 0:
            return

        output_dir = f"{base_path}_{step}"
        if self.is_world_process_zero():
            print(f"Saving named TrackGRPO checkpoint to: {output_dir}")

        # `save_model` is DeepSpeed-aware. All ranks should enter this call.
        self.save_model(output_dir)
        if self.is_world_process_zero():
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        if hasattr(self, "accelerator"):
            self.accelerator.wait_for_everyone()

    def _save_checkpoint(self, model, trial, metrics=None):
        """Hook Trainer's normal step checkpointing and add grpo_uav123_<step>."""
        try:
            super()._save_checkpoint(model, trial, metrics=metrics)
        except TypeError:
            # Older transformers versions do not have the `metrics` argument.
            super()._save_checkpoint(model, trial)
        self._save_named_model_checkpoint(int(self.state.global_step))

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        num_generations = int(self.grpo_config.get("num_generations", 4))
        clip_eps = float(self.grpo_config.get("clip_epsilon", 0.2))
        beta_kl = float(self.grpo_config.get("beta_kl", 0.0))

        with torch.no_grad():
            completion_token_ids, completion_texts = self._generate_completions(model, inputs)
            rewards, reward_metrics = self._compute_rewards(completion_texts, inputs)
            advantages, valid_group = self._group_advantages(rewards, group_size=num_generations)

        base_prompts = [p for p in inputs["prompt"] for _ in range(num_generations)]
        frames_rep, n_frames_rep = repeat_visual_batch(inputs["frames"], inputs["n_frames"], num_generations)
        input_ids, attention_mask, labels = self._build_full_sequences(
            base_prompts, completion_token_ids, device=inputs["frames"].device
        )

        with torch.no_grad():
            old_logps, token_mask = self._per_token_logps(model, input_ids, attention_mask, labels, frames_rep, n_frames_rep)
            old_logps = old_logps.detach()
            token_mask = token_mask.detach()

        new_logps, token_mask = self._per_token_logps(model, input_ids, attention_mask, labels, frames_rep, n_frames_rep)

        # PPO/GRPO clipped token objective.  old_logps is detached, so ratio has
        # gradients through new_logps even though both are from the current policy.
        ratio = torch.exp(new_logps - old_logps)
        adv = advantages.view(-1, 1).to(new_logps.device)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        pg_loss = -torch.minimum(unclipped, clipped)
        pg_loss = (pg_loss * token_mask).sum() / token_mask.sum().clamp_min(1.0)

        kl_loss = torch.zeros_like(pg_loss)
        if beta_kl > 0.0 and self.reference_model is not None:
            policy_base = unwrap_model(model)
            ref_base = unwrap_model(self.reference_model)

            # Match reference dtype/device to the policy model.
            # Use adapter dtype because the crash happens inside adapter.pre_proj.
            ref_dtype = next(policy_base.adapter.parameters()).dtype
            ref_device = input_ids.device

            if next(ref_base.parameters()).device != ref_device or next(ref_base.adapter.parameters()).dtype != ref_dtype:
                self.reference_model.to(device=ref_device, dtype=ref_dtype)

            ref_base = unwrap_model(self.reference_model)
            self.reference_model.eval()

            with torch.no_grad():
                ref_logps, _ = self._per_token_logps(
                    self.reference_model,
                    input_ids,
                    attention_mask,
                    labels,
                    frames_rep.to(device=ref_device),
                    n_frames_rep,
                    force_dtype=ref_dtype,
                )
            # Non-negative k3 approximate KL used in many RLHF implementations.
            log_ratio = ref_logps - new_logps
            per_token_kl = torch.exp(log_ratio) - log_ratio - 1.0
            kl_loss = (per_token_kl * token_mask).sum() / token_mask.sum().clamp_min(1.0)

        loss = pg_loss + beta_kl * kl_loss

        if self.state.global_step % max(1, int(self.args.logging_steps)) == 0:
            logs = {
                "grpo/reward": float(rewards.mean().detach().cpu()),
                "grpo/reward_std": float(rewards.view(-1, num_generations).std(dim=1, unbiased=False).mean().detach().cpu()),
                "grpo/valid_group_frac": float(valid_group.mean().detach().cpu()),
                "grpo/adv_abs": float(advantages.abs().mean().detach().cpu()),
                "grpo/pg_loss": float(pg_loss.detach().cpu()),
                "grpo/kl_loss": float(kl_loss.detach().cpu()),
            }
            for k, v in reward_metrics.items():
                logs[f"reward/{k}"] = float(v)
            self.log(logs)

        return (loss, None) if return_outputs else loss


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def set_trainable_by_name(model: ElysiumForCausalLM, trainable_keywords: Sequence[str]):
    """Optional PEFT-light control without adding LoRA dependencies.

    If grpo.trainable_keywords is non-empty, freeze everything except parameters
    whose names contain one of the provided substrings.
    """

    if not trainable_keywords:
        return
    for name, p in model.named_parameters():
        p.requires_grad = any(key in name for key in trainable_keywords)


def apply_freeze_policy(model: ElysiumForCausalLM, grpo_config: dict):
    cfg = edict(grpo_config or {})
    trainable_keywords = cfg.get("trainable_keywords", [])
    if trainable_keywords:
        set_trainable_by_name(model, trainable_keywords)
        return

    if cfg.get("freeze_visual_encoder", True):
        for p in model.visual_encoder.parameters():
            p.requires_grad = False
    if cfg.get("freeze_llm", False):
        for p in model.llm.parameters():
            p.requires_grad = False
    if cfg.get("freeze_adapter", False):
        for p in model.adapter.parameters():
            p.requires_grad = False
    if cfg.get("freeze_projector", False):
        for p in model.llm_proj.parameters():
            p.requires_grad = False


def load_elysium_from_config(model_args: ModelArguments, grpo_config: dict) -> ElysiumForCausalLM:
    model_config = edict(model_args.model or {})
    checkpoint_path = model_config.get(
        "pretrained_model_name_or_path",
        model_config.get("checkpoint_path", edict(grpo_config or {}).get("checkpoint", "checkpoints/elysium_7b")),
    )
    print(f"Loading Elysium checkpoint from: {checkpoint_path}")
    config = ElysiumConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = ElysiumForCausalLM.from_pretrained(checkpoint_path, config=config, trust_remote_code=True)
    apply_freeze_policy(model, grpo_config)
    return model


def maybe_load_reference_model(model_args: ModelArguments, grpo_config: dict) -> Optional[ElysiumForCausalLM]:
    cfg = edict(grpo_config or {})
    beta_kl = float(cfg.get("beta_kl", 0.0))

    if not cfg.get("use_reference_model", False):
        return None

    if beta_kl <= 0.0:
        print("WARNING: use_reference_model=True but beta_kl=0.0, so the reference model will not be used.")
        return None

    model_config = edict(model_args.model or {})
    ref_path = (
        cfg.get("reference_model_path")
        or model_config.get("pretrained_model_name_or_path")
        or model_config.get("checkpoint_path")
        or cfg.get("checkpoint", "checkpoints/elysium_7b")
    )

    print(f"Loading frozen reference model from: {ref_path}")

    config = ElysiumConfig.from_pretrained(ref_path, trust_remote_code=True)

    dtype_name = str(cfg.get("reference_torch_dtype", "bf16")).lower()
    if dtype_name in {"bf16", "bfloat16"}:
        torch_dtype = torch.bfloat16
    elif dtype_name in {"fp16", "float16"}:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    ref = ElysiumForCausalLM.from_pretrained(
        ref_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    return ref


if __name__ == "__main__":
    os.environ.setdefault("WANDB_PROJECT", "Elysium-GRPO")

    argument_parser = ArgumentParser()
    # Keep the Elysium style: one YAML file controls the run.
    argument_parser.add_argument("--config", type=str, required=True)
    # DeepSpeed/torchrun may inject this; no experiment setting should be passed here.
    argument_parser.add_argument("--local_rank", type=int, default=-1)
    args = argument_parser.parse_args()

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, GRPOArguments))
    model_args, data_args, training_args, grpo_args = parser.parse_yaml_file(args.config, allow_extra_keys=True)

    random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)

    df_config = edict(data_args.data).train.data_fetch
    dp_config = edict(data_args.data).train.data_preprocess
    dp_config.update({"meta_keys": ["source", "id", "question", "gt"]})
    processor = VideoLLMProcessor(**dp_config)

    train_dataset = LocalDataset(
        data_paths=df_config.data_paths,
        multi_round_qa=df_config.get("multi_round_qa", True),
        processor=processor,
    )

    # Copy GRPO config so we can inject model-level paths without mutating parser internals.
    grpo_config = dict(grpo_args.grpo or {})
    model_config = edict(model_args.model or {})
    if model_config.get("trained_model_name_or_path"):
        grpo_config["trained_model_name_or_path"] = model_config.get("trained_model_name_or_path")

    model = load_elysium_from_config(model_args, grpo_config)
    reference_model = maybe_load_reference_model(model_args, grpo_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters = {trainable} / {total}")

    trainer = TrackGRPOTrainer(
        model=model,
        reference_model=reference_model,
        processor=processor,
        grpo_config=grpo_config,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=processor.batch_transform,
    )
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

"""
CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=/raid/hvtham/dhviet/ElysiumGRPO/Elysium-main \
deepspeed --master_port=29691 training/train_grpo.py --config configs/sft_grpo_uav123_v2.yaml
"""
