
# scripts/int4_quantize.py
import argparse
import time

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from bench import run

MODEL_ID = "baidu/Unlimited-OCR-2"

parser = argparse.ArgumentParser()
parser.add_argument("--image", default="data/image.png")
parser.add_argument("--output", default="results/int4.json")
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
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

run(
    model, tokenizer, "int4_nf4", args.image, args.output, load_seconds, runs=args.runs,
    notes="bitsandbytes NF4 + double quant, all linear layers (no skip list)",
)
