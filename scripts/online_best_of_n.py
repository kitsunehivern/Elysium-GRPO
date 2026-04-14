import copy
import json
import math
import os
import os.path as osp
import re
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import transformers
from easydict import EasyDict as edict
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from data.processors.box_processor import BOX_PROCESSORS
from data.video_llm_data import VideoLLMPredictProcessor


global global_box_pool
global_box_pool = {}


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


def to_jsonable(obj):
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    return obj


def parse_box_from_raw_text(text, coords_pattern=r"{<(\d+)><(\d+)><(\d+)><(\d+)>}"):
    try:
        raw_coords = re.findall(coords_pattern, text)
        if len(raw_coords) < 1:
            raw_coords = re.findall(r"\[([\d\s,]+)\]", text)
            coords = []
            for xyxy_str in raw_coords:
                box = []
                for coord in xyxy_str.replace(" ", "").split(","):
                    box.append(float(coord) / 100)
                box = box[:4]
                if len(box) < 4:
                    box = coords[-1] if len(coords) > 0 else [0, 0, 0, 0]
                coords.append(box)
        else:
            coords = [[float(coord) / 100 for coord in xyxy_str][:4] for xyxy_str in raw_coords]
        return coords
    except Exception as e:
        print(e)
        return []


class LongVideoDistributedSampler(DistributedSampler):
    def __init__(self, start_indices, **kwargs) -> None:
        self.start_indices = start_indices
        super().__init__(**kwargs)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        iter_indices = []
        for i in range(self.rank, len(self.start_indices) - 1, self.num_replicas):
            start_index = self.start_indices[i]
            end_index = self.start_indices[i + 1]
            iter_indices.extend(indices[start_index:end_index])
        return iter(iter_indices)


class LocalDataset(Dataset):
    BOX_STR_TEMPLATE = "Frame {i}: <box>"
    FRAME_STR_TEMPLATE = "Frame {i}: <image>"

    SOT_QUESTION_TEMPLATE = (
        "{frame_str}This is a video showing an object with coordinates <box> in Frame 1. "
        "Please provide the detailed coordinates of the object in each frame."
    )
    RSOT_QUESTION_TEMPLATE = (
        "{frame_str}Please find one {object_class} and provide the detailed coordinates in each frame."
    )

    def __init__(self, image_folder, anno_path, clip_len=8, task="RSOT", processor=None):
        self.image_folder = image_folder
        self.processor = processor
        self.clip_len = clip_len
        self.task = task
        self.start_indices = []

        self.box_processor = BOX_PROCESSORS["ours_v1"]
        with open(anno_path, "r") as f:
            anns = [json.loads(line) for line in f]
        self.preprocess(anns)

    def preprocess(self, anns):
        self.anns = []
        for item in anns:
            self.start_indices.append(len(self.anns))
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
                clip_data = dict(
                    frames_path=clip_frames_path,
                    box=clip_boxes,
                    inital_box=clip_boxes[0],
                    frame_size=item["frame_size"],
                    object_description=item["object_description"],
                    object_class=item["object_class"],
                    video_folder=item["vid"],
                    seq_id=item["vid"],
                    clip_id=clip_id,
                )
                self.anns.append(clip_data)
        self.start_indices.append(len(self.anns))

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        item = copy.deepcopy(self.anns[idx])
        seq_id = item["seq_id"]
        clip_id = item["clip_id"]

        if clip_id == 0:
            initial_box = item["inital_box"]
        else:
            initial_box = global_box_pool[f"{seq_id}|{clip_id-1}"]

        frame_len = len(item["frames_path"])
        frame_str = ", ".join(
            self.FRAME_STR_TEMPLATE.format(**{"i": i + 1}) for i in range(frame_len)
        ) + "\n"
        box_str = ", ".join(
            self.BOX_STR_TEMPLATE.format(**{"i": i + 1}) for i in range(frame_len)
        )

        if self.task == "SOT":
            question = self.SOT_QUESTION_TEMPLATE.format(**{"frame_str": frame_str})
            answer = box_str
        elif self.task == "RSOT":
            question = self.RSOT_QUESTION_TEMPLATE.format(
                **{"frame_str": frame_str, "object_class": item["object_class"]}
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
                answer = self.box_processor(answer, [item["inital_box"]])
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
            "gt_boxes_raw": item["box"],
        }
        output = self.processor.transform(data_dict)
        output["gt_boxes_raw"] = item["box"]
        output["raw_question"] = question
        output["raw_gt"] = answer
        output["raw_image_size"] = item["frame_size"]
        return output


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union = box_area(box_a) + box_area(box_b) - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union

def canonicalize_gt_boxes(gt_boxes_raw):
    """
    Convert GT boxes into Python list of shape [T, 4].

    Accepts:
      - [T, 4]
      - [1, T, 4]
      - [T, 4, 1]
      - [4, T]
      - [4]   -> treated as one box [[x1,y1,x2,y2]]
    """
    import torch

    if isinstance(gt_boxes_raw, torch.Tensor):
        x = gt_boxes_raw.detach().cpu()
    else:
        x = torch.tensor(gt_boxes_raw, dtype=torch.float32)

    x = x.squeeze()

    if x.ndim == 1:
        if x.shape[0] == 4:
            return [x.tolist()]
        raise ValueError(f"Unsupported 1D gt_boxes_raw shape: {tuple(x.shape)}")

    if x.ndim == 2:
        if x.shape[-1] == 4:
            return x.tolist()
        if x.shape[0] == 4:
            return x.transpose(0, 1).tolist()

    if x.ndim == 3:
        x = x.squeeze()
        if x.ndim == 2:
            if x.shape[-1] == 4:
                return x.tolist()
            if x.shape[0] == 4:
                return x.transpose(0, 1).tolist()

    raise ValueError(f"Unsupported gt_boxes_raw shape: {tuple(x.shape)}")

def box_center(box):
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def center_distance(box_a, box_b):
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def normalize_box(box, image_size):
    w, h = image_size
    x1, y1, x2, y2 = box
    # gt in annotation is usually absolute pixels; parsed predictions are normalized [0,1]
    # convert gt to normalized [0,1]
    return [x1 / w, y1 / h, x2 / w, y2 / h]


def fix_pred_len(pred_boxes, target_len):
    pred_boxes = pred_boxes[:target_len]
    if len(pred_boxes) == 0:
        pred_boxes = [[0.0, 0.0, 0.0, 0.0] for _ in range(target_len)]
    while len(pred_boxes) < target_len:
        pred_boxes.append(pred_boxes[-1])
    return pred_boxes


def score_candidate(pred_boxes, gt_boxes_norm, image_size):
    gt_boxes = gt_boxes_norm
    pred_boxes = fix_pred_len(pred_boxes, len(gt_boxes))

    ious = []
    center_errors = []
    valid = 0

    for pb, gb in zip(pred_boxes, gt_boxes):
        print(f"GT box: {gb}, Pred box: {pb}")

        x1, y1, x2, y2 = pb
        is_valid = 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0

        if is_valid:
            valid += 1
            ious.append(box_iou(pb, gb))
            center_errors.append(center_distance(pb, gb))
        else:
            ious.append(0.0)
            center_errors.append(10.0)

    jitter = 0.0
    for i in range(1, len(pred_boxes)):
        jitter += center_distance(pred_boxes[i - 1], pred_boxes[i])
    if len(pred_boxes) > 1:
        jitter /= (len(pred_boxes) - 1)

    mean_iou = sum(ious) / len(ious)
    mean_center_error = sum(center_errors) / len(center_errors)
    format_valid_rate = valid / len(gt_boxes)

    score = (
        5.0 * mean_iou
        + 1.0 * format_valid_rate
        - 0.5 * mean_center_error
        - 0.2 * jitter
    )

    return {
        "score": score,
        "mean_iou": mean_iou,
        "mean_center_error": mean_center_error,
        "format_valid_rate": format_valid_rate,
        "jitter": jitter,
        "pred_boxes": pred_boxes,
    }


class VideoLLMBestOfNEvaluator:
    def __init__(self, model, data_args, task, num_candidates=4, temperature=0.8, top_p=0.95, **kwargs):
        super().__init__(**kwargs)
        self.data_args = edict(data_args.data)
        self.task = task
        self.num_candidates = num_candidates
        self.temperature = temperature
        self.top_p = top_p
        self.dataloader = self.get_dataloader(self.data_args.predict)
        self.model = model.cuda().eval()

    def get_dataloader(self, config):
        df_config = config.data_fetch
        dp_config = config.data_preprocess
        dp_config.update({"meta_keys": ["source", "id", "question", "gt"]})
        processor = VideoLLMPredictProcessor(**dp_config)

        dataset = LocalDataset(
            image_folder=df_config.image_folder,
            anno_path=df_config.anno_path,
            processor=processor,
            task=self.task,
        )

        sampler = LongVideoDistributedSampler(
            start_indices=dataset.start_indices,
            dataset=dataset,
        )
        loader = DataLoader(
            dataset,
            batch_size=df_config.batch_sizes[0],
            sampler=sampler,
            prefetch_factor=None,
            collate_fn=processor.batch_transform,
            num_workers=0,
            shuffle=False,
        )
        return loader

    def sample_candidates_for_single(self, batch: Dict[str, Any], sample_idx: int):
        """
        Extract one logical sample from the collated batch.

        Important:
        - frames is flattened across all frames in the batch
        - n_frames tells how many frames belong to each sample
        """
        single = {}

        # n_frames may be tensor/list; convert to python ints
        if isinstance(batch["n_frames"], torch.Tensor):
            n_frames_list = [int(x) for x in batch["n_frames"].detach().cpu().tolist()]
        else:
            n_frames_list = [int(x) for x in batch["n_frames"]]

        start = sum(n_frames_list[:sample_idx])
        end = start + n_frames_list[sample_idx]

        for key, value in batch.items():
            if key == "frames":
                # frames is flattened over all frames from all samples
                single[key] = value[start:end]

            elif key == "n_frames":
                # keep as one-sample tensor/list with the correct frame count
                if isinstance(value, torch.Tensor):
                    single[key] = value[sample_idx:sample_idx + 1]
                else:
                    single[key] = [value[sample_idx]]

            elif isinstance(value, torch.Tensor):
                # normal batched tensors: input_ids, attention_mask, etc.
                single[key] = value[sample_idx:sample_idx + 1]

            elif isinstance(value, list):
                single[key] = [value[sample_idx]]

            else:
                single[key] = value

        generate_params = dict(
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            max_new_tokens=2048,
            min_length=4,
            repetition_penalty=1.0,
            length_penalty=1.0,
        )

        # First try native multi-return generation
        try:
            outputs = self.model.generate(
                single["frames"],
                single["n_frames"],
                single["input_ids"],
                single["attention_mask"],
                num_return_sequences=self.num_candidates,
                **generate_params
            )
            return outputs if isinstance(outputs, list) else [outputs]

        except TypeError:
            # Fallback if custom generate does not support num_return_sequences
            pass

        # Safe fallback: sample one candidate at a time
        outputs = []
        for _ in range(self.num_candidates):
            pred = self.model.generate(
                single["frames"],
                single["n_frames"],
                single["input_ids"],
                single["attention_mask"],
                **generate_params
            )
            if isinstance(pred, list):
                outputs.extend(pred)
            else:
                outputs.append(pred)

        return outputs

    def predict(self, save_path):
        f = open(save_path, "a", encoding="utf-8")

        for _, batch in tqdm(enumerate(self.dataloader)):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cuda()

            batch_size = len(batch["id"])

            for i in range(batch_size):
                outputs = self.sample_candidates_for_single(batch, i)

                image_size = [
                    batch["image_size"][0][i].item(),
                    batch["image_size"][1][i].item()
                ]
                raw_gt = batch["gt_boxes_raw"]
                if isinstance(raw_gt, torch.Tensor):
                    print("[debug] batch['gt_boxes_raw'] shape:", tuple(raw_gt.shape))
                else:
                    print("[debug] batch['gt_boxes_raw'] type:", type(raw_gt))
                    try:
                        print("[debug] batch['gt_boxes_raw'] len:", len(raw_gt))
                        print("[debug] first element:", raw_gt[0])
                    except Exception:
                        print("[debug] batch['gt_boxes_raw'] value:", raw_gt)

                gt_boxes_norm = canonicalize_gt_boxes(raw_gt)
                print("[debug] num gt boxes:", len(gt_boxes_norm))
                print("[debug] first gt box:", gt_boxes_norm[0])

                candidate_infos = []
                for pred in outputs:
                    pred_boxes = parse_box_from_raw_text(pred)
                    metrics = score_candidate(pred_boxes, gt_boxes_norm, image_size)
                    candidate_infos.append({
                        "predict": pred,
                        **metrics,
                    })

                candidate_infos = sorted(candidate_infos, key=lambda x: x["score"], reverse=True)
                best = candidate_infos[0]

                global_box_pool[batch["id"][i]] = best["pred_boxes"][-1]

                line = {
                    "vid": batch["vid"][i],
                    "id": batch["id"][i],
                    "question": batch["question"][i],
                    "prompt": batch["prompt"][i] if "prompt" in batch else None,
                    "gt": batch["gt"][i],
                    "predict": best["predict"],
                    "image_sizes": image_size,
                    "best_of_n_meta": {
                        "num_candidates": self.num_candidates,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "score": best["score"],
                        "mean_iou": best["mean_iou"],
                        "mean_center_error": best["mean_center_error"],
                        "format_valid_rate": best["format_valid_rate"],
                        "jitter": best["jitter"],
                    },
                    "all_candidates": [
                        {
                            "predict": c["predict"],
                            "score": c["score"],
                            "mean_iou": c["mean_iou"],
                            "mean_center_error": c["mean_center_error"],
                            "format_valid_rate": c["format_valid_rate"],
                            "jitter": c["jitter"],
                        }
                        for c in candidate_infos
                    ],
                }

                line = to_jsonable(line)
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                f.flush()

                print(
                    f"[best-of-n] id={line['id']} "
                    f"score={best['score']:.4f} "
                    f"iou={best['mean_iou']:.4f} "
                    f"valid={best['format_valid_rate']:.4f}"
                )

        f.close()


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl")

    argument_parser = ArgumentParser()
    argument_parser.add_argument("--config", type=str, required=True)
    argument_parser.add_argument("--local_rank", type=int)
    argument_parser.add_argument("--task", type=str, choices=("SOT", "RSOT"), default="SOT")
    argument_parser.add_argument("--num_candidates", type=int, default=4)
    argument_parser.add_argument("--temperature", type=float, default=0.8)
    argument_parser.add_argument("--top_p", type=float, default=0.95)
    args = argument_parser.parse_args()

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_yaml_file(args.config)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model["model_name_or_path"],
        trust_remote_code=True
    )

    evaluator = VideoLLMBestOfNEvaluator(
        model=model,
        data_args=data_args,
        task=args.task,
        num_candidates=args.num_candidates,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    save_filename = osp.basename(edict(data_args.data).predict.data_fetch.anno_path)
    save_folder = osp.join(training_args.output_dir, f"best_of_n_{args.task.lower()}")
    save_path = osp.join(save_folder, save_filename)
    os.makedirs(save_folder, exist_ok=True)

    evaluator.predict(save_path=save_path)

"""
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/raid/hvtham/dcmquan/Elysium \
deepspeed scripts/online_best_of_n.py \
  --config configs/baseline.yaml \
  --task SOT \
  --num_candidates 4 \
  --temperature 0.8 \
  --top_p 0.95
"""