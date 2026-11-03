
# scripts/bench.py
# Shared measurement for bf16 / int8 / int4 runs, so every variant is timed,
# measured and scored the same way.
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

GB = 1024**3


def strip_layout(text):
    """Remove <PAGE>/<BLOCK>type<LEVELn>[x1, y1, x2, y2]<FORK> markup, keep only text."""
    text = re.sub(r"<BLOCK>[^\[<]*(?:<LEVEL\d+>)?\[[^\]]*\]<FORK>", "\n", text)
    text = re.sub(r"</?[A-Z]+\d*>", "", text)
    return normalize(text)


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def run(model, tokenizer, variant, image, output, load_seconds, warmup=2, runs=3, notes=None):
    """Warm up, time `runs` inferences, then save metrics to `output` (a .json path)."""
    weights_gb = model.get_memory_footprint() / GB
    allocated_after_load_gb = torch.cuda.memory_allocated() / GB

    def infer():
        # eval_mode=True turns off token streaming to stdout, which would skew latency
        return model.infer(tokenizer, image_file=image, save_results=False, eval_mode=True)

    for _ in range(warmup):
        infer()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    latencies = []
    predictions = []
    for _ in range(runs):
        start = time.perf_counter()
        predictions.append(str(infer()))
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)

    prediction = predictions[-1]
    output_tokens = len(tokenizer.encode(prediction, add_special_tokens=False))
    median_latency = statistics.median(latencies)

    result = {
        "variant": variant,
        "image": str(image),
        "notes": notes,
        "load_seconds": load_seconds,
        "memory": {
            "weights_gb": weights_gb,
            "allocated_after_load_gb": allocated_after_load_gb,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / GB,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / GB,
        },
        "speed": {
            "warmup_runs": warmup,
            "runs": runs,
            "latency_seconds": latencies,
            "latency_median_seconds": median_latency,
            "output_tokens": output_tokens,
            "tokens_per_second": output_tokens / median_latency,
        },
        "accuracy": None,
        "deterministic": len(set(predictions)) == 1,
        "prediction": prediction,
        "environment": environment(),
    }

    # Score against ground truth next to the image (data/image.png -> data/image.txt)
    ground_truth = Path(image).with_suffix(".txt")
    if ground_truth.exists():
        import jiwer

        reference = normalize(ground_truth.read_text(encoding="utf-8"))
        hypothesis = strip_layout(prediction)
        result["accuracy"] = {
            "ground_truth": str(ground_truth),
            "cer": jiwer.cer(reference, hypothesis),
            "wer": jiwer.wer(reference, hypothesis),
        }

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    summary = {k: result[k] for k in ("variant", "load_seconds", "memory", "speed", "accuracy", "deterministic")}
    print(json.dumps(summary, indent=2))
    print(f"Saved {output}")
    return result


def environment():
    try:
        import bitsandbytes

        bnb_version = bitsandbytes.__version__
    except ImportError:
        bnb_version = None
    return {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "bitsandbytes": bnb_version,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
