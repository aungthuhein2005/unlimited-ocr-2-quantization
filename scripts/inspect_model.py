from huggingface_hub import snapshot_download
from transformers import AutoModel
import torch
from collections import Counter
from pathlib import Path
import json

MODEL_ID = "baidu/Unlimited-OCR-2"
# Number of leading name components used to group layers (e.g. "model.vision_tower")
GROUP_DEPTH = 2
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

path = snapshot_download(MODEL_ID)

print(f"Model ID: {MODEL_ID}")
print(f"Model path: {path}")

# Loads the full model in bf16 on CPU: needs roughly 2 bytes per parameter of RAM
model = AutoModel.from_pretrained(
    path,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True)

linear_layers = [
    (name, module) for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
]

groups = Counter(
    ".".join(name.split(".")[:GROUP_DEPTH]) for name, _ in linear_layers
)

print("Linear layers:", len(linear_layers))
print(f"Groups (depth {GROUP_DEPTH}):")
for group, count in groups.most_common():
    print(f"  {group}: {count}")

for name, module in linear_layers:
    print(
        name,
        module.in_features,
        module.out_features,
    )

RESULTS_DIR.mkdir(exist_ok=True)
out_path = RESULTS_DIR / "layer_inventory.json"
with open(out_path, "w") as f:
    json.dump(
        [
            {
                "name": name,
                "in_features": m.in_features,
                "out_features": m.out_features,
            }
            for name, m in linear_layers
        ],
        f,
        indent=2,
    )
print(f"Wrote {out_path}")
