
# scripts/bench.py
# Shared measurement for bf16 / int8 / int4 runs, so every variant is timed,
# measured and scored the same way.
import glob
import json
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

GB = 1024**3


def add_image_args(parser, output):
    parser.add_argument(
        "--images", nargs="+", default=["data/*.png"],
        help="image paths or glob patterns (default: data/*.png)",
    )
    parser.add_argument("--output", default=output)
    parser.add_argument("--warmup", type=int, default=1, help="untimed runs per image")
    parser.add_argument("--runs", type=int, default=3, help="timed runs per image")


def expand_images(patterns):
    images = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) or [pattern]
        images.extend(m for m in matches if m not in images)
    missing = [i for i in images if not Path(i).is_file()]
    if missing:
        raise FileNotFoundError(f"Images not found: {missing}")
    return images


def strip_layout(text):
    """Remove <PAGE>/<BLOCK>type<LEVELn>[x1, y1, x2, y2]<FORK> markup, keep only text."""
    text = re.sub(r"<BLOCK>[^\[<]*(?:<LEVEL\d+>)?\[[^\]]*\]<FORK>", "\n", text)
    text = re.sub(r"</?[A-Z]+\d*>", "", text)
    return normalize(text)


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def run(model, tokenizer, variant, images, output, load_seconds, warmup=1, runs=3, notes=None):
    """Benchmark every image, then save per-image metrics and a summary to `output` (.json)."""
    images = expand_images(images)
    print(f"Benchmarking {len(images)} image(s): {images}")

    result = {
        "variant": variant,
        "notes": notes,
        "load_seconds": load_seconds,
        "memory": {
            "weights_gb": model.get_memory_footprint() / GB,
            "allocated_after_load_gb": torch.cuda.memory_allocated() / GB,
        },
        "settings": {"warmup_runs_per_image": warmup, "timed_runs_per_image": runs},
        "summary": None,
        "images": [run_image(model, tokenizer, image, warmup, runs) for image in images],
        "environment": environment(),
    }
    result["summary"] = summarize(result["images"])

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: result[k] for k in ("variant", "load_seconds", "memory", "summary")}, indent=2))
    print(f"Saved {output}")
    return result


def run_image(model, tokenizer, image, warmup, runs):
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

    entry = {
        "image": str(image),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / GB,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / GB,
        "latency_seconds": latencies,
        "latency_median_seconds": median_latency,
        "output_tokens": output_tokens,
        "tokens_per_second": output_tokens / median_latency,
        "accuracy": None,
        "deterministic": len(set(predictions)) == 1,
        "prediction": prediction,
    }

    # Score against ground truth next to the image (data/en_image.png -> data/en_image.txt)
    ground_truth = Path(image).with_suffix(".txt")
    if ground_truth.exists():
        import jiwer

        reference = normalize(ground_truth.read_text(encoding="utf-8"))
        hypothesis = strip_layout(prediction)
        entry["accuracy"] = {
            "ground_truth": str(ground_truth),
            "cer": jiwer.cer(reference, hypothesis),
            "wer": jiwer.wer(reference, hypothesis),
        }

    acc = entry["accuracy"]
    score = f"CER {acc['cer']:.2%} WER {acc['wer']:.2%}" if acc else "no ground truth"
    print(f"{image}: {median_latency:.2f}s, {entry['tokens_per_second']:.1f} tok/s, "
          f"peak {entry['peak_allocated_gb']:.2f} GB, {score}")
    return entry


def summarize(entries):
    scored = [e["accuracy"] for e in entries if e["accuracy"]]
    total_tokens = sum(e["output_tokens"] for e in entries)
    total_seconds = sum(e["latency_median_seconds"] for e in entries)
    return {
        "images": len(entries),
        "images_with_ground_truth": len(scored),
        "peak_allocated_gb_max": max(e["peak_allocated_gb"] for e in entries),
        "latency_median_seconds_mean": total_seconds / len(entries),
        # Overall throughput: all tokens over all (median) time, so long pages weigh more
        "tokens_per_second": total_tokens / total_seconds,
        "cer_mean": statistics.mean(a["cer"] for a in scored) if scored else None,
        "wer_mean": statistics.mean(a["wer"] for a in scored) if scored else None,
        "all_deterministic": all(e["deterministic"] for e in entries),
    }


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
