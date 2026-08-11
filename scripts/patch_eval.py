with open("/home/stackops/dhviet/Elysium-GRPO/eval/eval.py", "r") as f:
    script = f.read()

script = "from transformers import BitsAndBytesConfig\nimport torch\n" + script

script = script.replace(
    'model = AutoModelForCausalLM.from_pretrained("elysium_7b"',
    'model = AutoModelForCausalLM.from_pretrained("/home/stackops/dhviet/Elysium-GRPO/checkpoints/elysium_7b"'
)

script = script.replace(
    "self.model = model.cuda().eval()",
    "self.model = model.eval()"
)

script = script.replace(
    "f.write(json.dumps(line, ensure_ascii=False) + '\\n')",
    """f.write(json.dumps({
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in line.items()
            }, ensure_ascii=False) + '\\n')"""
)


with open("/home/stackops/dhviet/Elysium-GRPO/eval/eval.py", "w") as f:
    f.write(script)

print("Patched model path in eval.py")