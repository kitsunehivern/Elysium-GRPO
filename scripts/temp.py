import json
with open("../../UAV123_Elysium/short_train/annotation.jsonl") as f:
    item = json.loads(f.readline())

vqa = item["vqa"] if isinstance(item["vqa"], list) else json.loads(item["vqa"])
print(vqa[0]["value"][:200])          # how many <image> tokens?
print(item["frames"][:3])             # how many total frames?
print(len(item["frames"]))