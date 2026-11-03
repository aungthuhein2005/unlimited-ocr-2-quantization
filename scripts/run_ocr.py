
# scripts/run_ocr.py
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "baidu/Unlimited-OCR-2"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="results/bf16")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()

    print(f"GPU: {torch.cuda.get_device_name()}")

    # Warm up before measured requests.
    # eval_mode=True turns off token streaming to stdout, which would skew latency.
    for _ in range(2):
        model.infer(
            tokenizer,
            image_file=args.image,
            output_path=str(output_dir),
            save_results=False,
            eval_mode=True,
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    prediction = model.infer(
        tokenizer,
        image_file=args.image,
        output_path=str(output_dir),
        save_results=False,
        eval_mode=True,
    )

    torch.cuda.synchronize()
    latency = time.perf_counter() - start
    peak_gb = (
        torch.cuda.max_memory_allocated() / 1024**3
    )

    result = {
        "prediction": str(prediction),
        "latency_seconds": latency,
        "peak_allocated_gb": peak_gb,
    }

    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
