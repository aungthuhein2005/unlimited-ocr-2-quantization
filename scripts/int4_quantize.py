
# scripts/int4_quantize.py
import argparse
import time

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from bench import add_image_args, run

MODEL_ID = "baidu/Unlimited-OCR-2"
# Same skip list as int8_quantize.py, for a like-for-like comparison
VISION_MODULES = ["sam_model", "vision_model", "projector", "lm_head"]

parser = argparse.ArgumentParser()
add_image_args(parser, "results/int4.json")
parser.add_argument(
    "--skip-vision", action="store_true",
    help=f"keep {VISION_MODULES} in bf16 (default output: results/int4_skipvision.json)",
)
args = parser.parse_args()
if args.skip_vision and args.output == "results/int4.json":
    args.output = "results/int4_skipvision.json"

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    # llm_int8_skip_modules also applies to 4-bit loading
    llm_int8_skip_modules=VISION_MODULES if args.skip_vision else None,
)

start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=True
)

model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
).eval()

torch.cuda.synchronize()
load_seconds = time.perf_counter() - start

print("Loaded NF4 candidate")

if args.skip_vision:
    variant = "int4_nf4_skipvision"
    notes = f"bitsandbytes NF4 + double quant, skipped (bf16): {VISION_MODULES}"
else:
    variant = "int4_nf4"
    notes = "bitsandbytes NF4 + double quant, all linear layers (no skip list)"

run(
    model, tokenizer, variant, args.images, args.output, load_seconds,
    warmup=args.warmup, runs=args.runs, notes=notes,
)
