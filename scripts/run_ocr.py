
# scripts/run_ocr.py
# bf16 baseline
import argparse
import time

import torch
from transformers import AutoModel, AutoTokenizer

from bench import add_image_args, run

MODEL_ID = "baidu/Unlimited-OCR-2"


def main():
    parser = argparse.ArgumentParser()
    add_image_args(parser, "results/bf16.json")
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

    run(
        model, tokenizer, "bf16", args.images, args.output, load_seconds,
        warmup=args.warmup, runs=args.runs,
    )


if __name__ == "__main__":
    main()
