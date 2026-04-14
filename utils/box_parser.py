import re
from typing import List, Optional

BOX_PATTERN = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")

def parse_boxes_from_text(text: str) -> List[List[int]]:
    boxes = []
    for match in BOX_PATTERN.finditer(text):
        x1, y1, x2, y2 = map(int, match.groups())
        boxes.append([x1, y1, x2, y2])
    return boxes

def clip_or_pad_boxes(
    boxes: List[List[int]],
    target_len: int,
    pad_with_last: bool = True,
) -> List[Optional[List[int]]]:
    if len(boxes) >= target_len:
        return boxes[:target_len]

    if not boxes:
        return [None] * target_len

    result = boxes[:]
    while len(result) < target_len:
        result.append(result[-1] if pad_with_last else None)
    return result
