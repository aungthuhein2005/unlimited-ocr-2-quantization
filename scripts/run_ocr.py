
# scripts/run_ocr.py
# bf16 baseline
import argparse
import time

import torch
from transformers import AutoModel, AutoTokenizer

from bench import run

MODEL_ID = "baidu/Unlimited-OCR-2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="data/image.png")
    parser.add_argument("--output", default="results/bf16.json")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()

    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start

    print(f"GPU: {torch.cuda.get_device_name()}")

    run(model, tokenizer, "bf16", args.image, args.output, load_seconds, runs=args.runs)


if __name__ == "__main__":
    main()
