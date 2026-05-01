from reward.grpo_reward import parse_box_from_raw_text, compute_sequence_reward


def show(name, text, gt):
    pred = parse_box_from_raw_text(text)
    info = compute_sequence_reward(
        pred_boxes_norm=pred,
        gt_boxes=gt,
        image_size=[640, 480],
        ignore_first_frame=True,
    )

    print("\n===", name, "===")
    print("pred boxes:", pred)
    for k in [
        "sequence_reward",
        "mean_iou",
        "complete",
        "complete_rate",
        "format_valid_rate",
        "mean_center_error",
        "mean_scale_error",
        "mean_jitter",
        "extra_rate",
        "num_pred_boxes",
        "num_gt_boxes",
    ]:
        print(k, "=", info[k])


def main():
    gt = [
        [0.10, 0.10, 0.20, 0.20],
        [0.11, 0.10, 0.21, 0.20],
        [0.12, 0.10, 0.22, 0.20],
        [0.13, 0.10, 0.23, 0.20],
    ]

    good = (
        "Frame 1: [10,10,20,20], "
        "Frame 2: [11,10,21,20], "
        "Frame 3: [12,10,22,20], "
        "Frame 4: [13,10,23,20]"
    )

    one_box = "Frame 1: [10,10,20,20]"
    empty = ""

    show("good", good, gt)
    show("one_box", one_box, gt)
    show("empty", empty, gt)


if __name__ == "__main__":
    main()

"""
PYTHONPATH=/raid/hvtham/dcmquan/Elysium \
python scripts/check_grpo_reward.py
"""