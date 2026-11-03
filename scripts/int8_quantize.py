
# scripts/int8_quantize.py
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
SKIP_MODULES = ["sam_model", "vision_model", "projector", "lm_head"]

parser = argparse.ArgumentParser()
parser.add_argument("--image", default="data/image.png")
parser.add_argument("--output", default="results/int8.json")
parser.add_argument("--runs", type=int, default=3)
args = parser.parse_args()

config = BitsAndBytesConfig(
    load_in_8bit=True,
    # Keep the vision side in bf16: SAM feeds 4D tensors to its linears,
    # which bitsandbytes' int8 matmul rejects (2D/3D only).
    llm_int8_skip_modules=SKIP_MODULES,
)

start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
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

print("Loaded INT8 candidate")

run(
    model, tokenizer, "int8", args.image, args.output, load_seconds, runs=args.runs,
    notes=f"bitsandbytes LLM.int8, skipped (bf16): {SKIP_MODULES}",
)
