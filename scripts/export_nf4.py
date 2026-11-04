
# scripts/export_nf4.py
# Save Unlimited-OCR-2 in bitsandbytes NF4 (same config as int4_quantize.py), make the folder
# self-contained (model code + model card), then reload it and check it reads a page the same way.
# Upload afterwards yourself:
#   huggingface-cli login
#   huggingface-cli upload <repo-id> export/Unlimited-OCR-2-bnb-nf4 .
import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)

MODEL_ID = "baidu/Unlimited-OCR-2"
GB = 1024**3

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="export/Unlimited-OCR-2-bnb-nf4")
parser.add_argument("--repo-id", default="aungthuhein-dev/Unlimited-OCR-2-bnb-nf4",
                    help="Hugging Face repo name used in the model card's usage example")
parser.add_argument("--test-image", default="data/en_image.png")
args = parser.parse_args()

output = Path(args.output)
if output.exists():
    raise SystemExit(f"{output} already exists; remove it or pass --output")

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 1. Quantize and save
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
).eval()
print(f"Loaded NF4, weights {model.get_memory_footprint() / GB:.2f} GB")

model.save_pretrained(output, safe_serialization=True)
tokenizer.save_pretrained(output)
del model
gc.collect()
torch.cuda.empty_cache()

# 2. Copy the model's own code, so trust_remote_code loads it from this repo
source = Path(snapshot_download(MODEL_ID))
for code_file in sorted(source.glob("*.py")):
    if not (output / code_file.name).exists():
        shutil.copy2(code_file, output / code_file.name)
        print(f"copied {code_file.name}")

# auto_map may point back to the original repo ("baidu/Unlimited-OCR-2--module.Class");
# keep only "module.Class" so the export does not depend on it
for name in ("config.json", "tokenizer_config.json"):
    path = output / name
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    auto_map = data.get("auto_map", {})
    for key, value in auto_map.items():
        values = value if isinstance(value, list) else [value]
        values = [v.split("--", 1)[1] if isinstance(v, str) and "--" in v else v for v in values]
        auto_map[key] = values if isinstance(value, list) else values[0]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

saved = json.loads((output / "config.json").read_text(encoding="utf-8"))
assert "quantization_config" in saved, "config.json has no quantization_config"
assert "auto_map" in saved, "config.json has no auto_map"


# 3. Model card with benchmark numbers from results/*.json
def summary(variant):
    path = Path(f"results/{variant}.json")
    return json.loads(path.read_text(encoding="utf-8"))["summary"] if path.exists() else None


bf16, nf4 = summary("bf16"), summary("int4")
if bf16 and nf4:
    results = f"""## Benchmark

Measured on an NVIDIA H200 MIG `3g.71gb` slice, {nf4['images']} pages
({nf4['images_with_ground_truth']} with ground truth), 1 warm-up + 3 timed runs per page.

| | bf16 (original) | NF4 (this repo) |
|:--|--:|--:|
| Peak GPU memory | {bf16['peak_allocated_gb_max']:.2f} GB | **{nf4['peak_allocated_gb_max']:.2f} GB** ({nf4['peak_allocated_gb_max'] / bf16['peak_allocated_gb_max'] - 1:+.0%}) |
| Time per page | {bf16['latency_median_seconds_mean']:.2f} s | {nf4['latency_median_seconds_mean']:.2f} s ({nf4['latency_median_seconds_mean'] / bf16['latency_median_seconds_mean'] - 1:+.0%}) |
| Tokens / second | {bf16['tokens_per_second']:.1f} | {nf4['tokens_per_second']:.1f} |
| Mean CER | {bf16['cer_mean']:.3%} | {nf4['cer_mean']:.3%} |
| Mean WER | {bf16['wer_mean']:.3%} | {nf4['wer_mean']:.3%} |

The test set is small, so CER differences of a few characters are not significant.
"""
else:
    results = "## Benchmark\n\nNo benchmark results were found when this card was generated.\n"

card = f"""---
license: mit
base_model: baidu/Unlimited-OCR-2
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- ocr
- bitsandbytes
- nf4
- 4-bit
---

# Unlimited-OCR-2 bnb NF4

4-bit [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) NF4 version of
[baidu/Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2).

- **Method:** bitsandbytes NF4, double quantization, bf16 compute
- **Quantized:** all linear layers (language model, MoE experts, vision encoders)
- **Calibration:** none (data-free)

{results}
## Requirements

- NVIDIA GPU with CUDA (bitsandbytes 4-bit does not run on CPU or Apple Silicon)
- `torch==2.13.0` (newer torch ships Triton 3.9, which breaks the model's attention kernels)
- `transformers==4.52.1`, `bitsandbytes`, `torchvision`, `einops`, `addict`, `easydict`

## Usage

```python
from transformers import AutoModel, AutoTokenizer

repo = "{args.repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModel.from_pretrained(repo, trust_remote_code=True, device_map="auto").eval()

text = model.infer(tokenizer, image_file="page.png", eval_mode=True)
print(text)
```

No `quantization_config` is needed: the settings are stored in `config.json`.

## License and attribution

The original model is released by Baidu under the MIT license; this quantized copy keeps that
license. This is an independent quantization and is not affiliated with or endorsed by Baidu.
"""
(output / "README.md").write_text(card, encoding="utf-8")

size = sum(f.stat().st_size for f in output.rglob("*") if f.is_file()) / GB
print(f"Saved {output} ({size:.2f} GB on disk)")

# 4. Reload from the export folder only (no quantization_config) and read one page
tokenizer = AutoTokenizer.from_pretrained(output, trust_remote_code=True)
model = AutoModel.from_pretrained(output, trust_remote_code=True, device_map="auto").eval()
print(f"Reloaded, weights {model.get_memory_footprint() / GB:.2f} GB")

prediction = str(model.infer(tokenizer, image_file=args.test_image, save_results=False, eval_mode=True))

reference = None
if Path("results/int4.json").exists():
    for entry in json.loads(Path("results/int4.json").read_text(encoding="utf-8"))["images"]:
        if Path(entry["image"]).as_posix() == Path(args.test_image).as_posix():
            reference = entry["prediction"]

if reference is None:
    print("No saved NF4 prediction for this image; output:")
    print(prediction)
elif prediction == reference:
    print(f"OK: reloaded export reads {args.test_image} exactly like the benchmark NF4 run")
else:
    raise SystemExit(f"MISMATCH on {args.test_image}: export output differs from the benchmark NF4 run")
