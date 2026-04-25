import copy
import json
import math
import os
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import transformers
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer import is_sagemaker_mp_enabled

from data.video_llm_data import VideoLLMProcessor
from data.processors.box_processor import BOX_PROCESSORS
from models.modeling_elysium import ElysiumForCausalLM, ElysiumConfig


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
    visual_encoder_lr_scale: int = field(default=1.0)
    remove_unused_columns: bool = field(default=False)
    using_torch_lr: bool = field(default=False)
    lr_type: str = field(default="")


class TrackingSFTDataset(Dataset):
    BOX_STR_TEMPLATE = "Frame {i}: <box>"
    FRAME_STR_TEMPLATE = "Frame {i}: <image>"

    SOT_QUESTION_TEMPLATE = (
        "{frame_str}This is a video showing an object with coordinates <box> in Frame 1. "
        "Please provide the detailed coordinates of the object in each frame."
    )

    RSOT_QUESTION_TEMPLATE = (
        "{frame_str}Please find one {object_class} and provide the detailed coordinates in each frame."
    )

    def __init__(self, data_paths, processor=None, task="SOT", clip_len=8):
        self.processor = processor
        self.task = task
        self.clip_len = clip_len
        self.box_processor = BOX_PROCESSORS["ours_v1"]
        self.anns = []

        for data_path in data_paths:
            image_folder = data_path["image_folder"]
            anno_path = data_path["anno_path"]

            with open(anno_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    item["image_folder"] = image_folder
                    self._preprocess_record(item)

    def _normalize_frame_path(self, p: str) -> str:
        # while p.startswith("../"):
        #     p = p[3:]
        # while p.startswith("./"):
        #     p = p[2:]
        return p

    def _preprocess_record(self, item):
        frames = item["frames"]
        boxes = item["box"]

        if len(frames) != len(boxes):
            raise ValueError(f"len(frames) != len(boxes) for {item.get('vid')}")

        stride = self.clip_len - 1
        n_clip = math.ceil(len(frames) / stride)

        for clip_id in range(n_clip):
            clip_frames = frames[clip_id * stride: clip_id * stride + self.clip_len]
            clip_boxes = boxes[clip_id * stride: clip_id * stride + self.clip_len]

            if len(clip_frames) < 2:
                continue

            clip_frames = [self._normalize_frame_path(p) for p in clip_frames]

            self.anns.append(
                {
                    "vid": item["vid"],
                    "clip_id": clip_id,
                    "frames": clip_frames,
                    "box": clip_boxes,
                    "frame_size": item["frame_size"],
                    "object_description": item.get("object_description", ""),
                    "object_class": item.get("object_class", "object"),
                    "image_folder": item["image_folder"],
                }
            )

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        item = copy.deepcopy(self.anns[idx])

        frame_len = len(item["frames"])
        initial_box = item["box"][0]

        frame_str = ", ".join(
            self.FRAME_STR_TEMPLATE.format(i=i + 1) for i in range(frame_len)
        ) + "\n"
        box_str = ", ".join(
            self.BOX_STR_TEMPLATE.format(i=i + 1) for i in range(frame_len)
        )

        if self.task == "SOT":
            question = self.SOT_QUESTION_TEMPLATE.format(frame_str=frame_str)
            answer = box_str
        elif self.task == "RSOT":
            question = self.RSOT_QUESTION_TEMPLATE.format(
                frame_str=frame_str,
                object_class=item["object_class"],
            )
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
                answer = self.box_processor(answer, [initial_box])
            else:
                answer = self.box_processor(answer, item["box"])

        messages = [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ]

        sample = {
            "id": f"{item['vid']}|{item['clip_id']}",
            "vid": item["vid"],
            "frames": item["frames"],
            "image_folder": item["image_folder"],
            "image_size": item["frame_size"],
            "question": question,
            "gt": answer,
            "vqa": messages,
        }

        try:
            return self.processor.transform(sample)
        except Exception as e:
            print("\n===== BAD SAMPLE =====")
            print("idx:", idx)
            print("keys:", list(sample.keys()))
            print("sample preview:", json.dumps(sample, ensure_ascii=False)[:3000])
            print("error:", repr(e))
            raise


class VideoLLMTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            scale_lr_parameters = [
                p for n, p in opt_model.named_parameters()
                if (n.startswith("visual_encoder") and p.requires_grad)
            ]

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (
                            n in decay_parameters
                            and not n.startswith("visual_encoder")
                            and p.requires_grad
                        )
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters()
                        if (
                            n not in decay_parameters
                            and not n.startswith("visual_encoder")
                            and p.requires_grad
                        )
                    ],
                    "weight_decay": 0.0,
                },
            ]

            if len(scale_lr_parameters) > 0:
                optimizer_grouped_parameters.append(
                    {
                        "params": scale_lr_parameters,
                        "weight_decay": 0.0,
                        "lr": self.args.visual_encoder_lr_scale * self.args.learning_rate,
                    }
                )

            optimizer_grouped_parameters = [
                group for group in optimizer_grouped_parameters if len(group["params"]) > 0
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        print(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        print(f"bitsandbytes: will optimize {module} in fp32")
                print(f"skipped: {skipped/2**20}M params")

        return self.optimizer


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


if __name__ == "__main__":
    global local_rank
    os.environ["WANDB_PROJECT"] = "Elysium"

    argument_parser = ArgumentParser()
    argument_parser.add_argument("--config", type=str, required=True)
    argument_parser.add_argument("--local_rank", type=int, default=-1)
    argument_parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    argument_parser.add_argument("--task", type=str, choices=["SOT", "RSOT"], default="SOT")
    argument_parser.add_argument("--clip_len", type=int, default=8)
    argument_parser.add_argument("--save_steps", type=int, default=None)
    args = argument_parser.parse_args()

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_yaml_file(args.config)

    local_rank = args.local_rank
    compute_dtype = (
        torch.float16 if training_args.fp16 else
        (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # Override save policy from CLI if requested
    training_args.save_strategy = "steps"
    if args.save_steps is not None:
        training_args.save_steps = args.save_steps
    elif getattr(training_args, "save_steps", None) is None:
        training_args.save_steps = 500

    df_config = edict(data_args.data).train.data_fetch
    dp_config = edict(data_args.data).train.data_preprocess
    dp_config.update({"meta_keys": ["source", "id", "question", "gt"]})
    processor = VideoLLMProcessor(**dp_config)

    train_dataset = TrackingSFTDataset(
        data_paths=df_config.data_paths,
        processor=processor,
        task=args.task,
        clip_len=args.clip_len,
    )

    resume_ckpt = args.resume_from_checkpoint
    model_path = model_args.model.get("model_name_or_path", "models")

    if resume_ckpt is not None:
        print(f"Loading model from resume checkpoint: {resume_ckpt}")
        model = ElysiumForCausalLM.from_pretrained(
            resume_ckpt,
            trust_remote_code=True,
        )
    else:
        print(f"Loading model from config model_name_or_path: {model_path}")
        try:
            model = ElysiumForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
        except Exception:
            print("from_pretrained failed, falling back to config initialization")
            config = ElysiumConfig.from_pretrained(model_path, trust_remote_code=True)
            model = ElysiumForCausalLM(config)

    if compute_dtype == torch.bfloat16:
        model = model.bfloat16()
    elif compute_dtype == torch.float16:
        model = model.half()
    else:
        model = model.float()

    print(f"Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"save_strategy = {training_args.save_strategy}")
    print(f"save_steps = {training_args.save_steps}")

    trainer = VideoLLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=processor.batch_transform,
    )

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_state()
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
    )

"""
CUDA_VISIBLE_DEVICES=5 \
PYTHONPATH=/raid/hvtham/dcmquan/Elysium \
deepspeed training/train.py \
  --config configs/sft_sot.yaml \
  --task SOT \
  --clip_len 8 \
  --save_steps 500
"""
