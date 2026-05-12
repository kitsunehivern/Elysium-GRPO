import re

paths = [
    # "/root/.cache/huggingface/modules/transformers_modules/elysium_7b/modeling_elysium.py",
    "/raid/hvtham/dhviet/ElysiumGRPO/Elysium-main/checkpoints/elysium_7b/modeling_elysium.py"
]

for path in paths:
    with open(path, "r") as f:
        code = f.read()

    code = re.sub(
        r'(class ElysiumForCausalLM\([^)]+\):)',
        r'\1\n    _no_split_modules = ["LlamaDecoderLayer"]',
        code
    )

    code = code.replace(
        '''    def _concat_embedding(self, vision_encode_out, input_ids, attention_mask, labels=None, left_padding=False):
        """ concat vision and text
        """
        vision_embeds, vision_atts, vision_targets = vision_encode_out''',

        '''    def _concat_embedding(self, vision_encode_out, input_ids, attention_mask, labels=None, left_padding=False):
        """ concat vision and text
        """
        # Normalize all tensors to the same device
        ref_device = next(self.llm.parameters()).device
        def to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(ref_device)
            elif isinstance(x, list):
                return [t.to(ref_device) if isinstance(t, torch.Tensor) else t for t in x]
            return x
        vision_embeds, vision_atts, vision_targets = vision_encode_out
        vision_embeds = to_device(vision_embeds)
        vision_atts = to_device(vision_atts)
        vision_targets = to_device(vision_targets)
        input_ids = input_ids.to(ref_device)
        attention_mask = attention_mask.to(ref_device)
        if labels is not None:
            labels = labels.to(ref_device)'''
    )

    with open(path, "w") as f:
        f.write(code)
    print(f"✓ Patched: {path}")

# !find / -name "*.pyc" -not -path "*/proc/*" -delete 2>/dev/null
# !find / -name "__pycache__" -not -path "*/proc/*" -exec rm -rf {} + 2>/dev/null

# Verify
# !grep -n "_no_split_modules\|class ElysiumForCausalLM" {model_path}