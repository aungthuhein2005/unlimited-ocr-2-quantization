<div align="center">

<img src="https://huggingface.co/baidu/Unlimited-OCR-2/resolve/main/assets/logo.png" alt="Unlimited-OCR-2 logo" width="160">

# unlimited-ocr-2-quantization

**Benchmarking bitsandbytes quantization of [baidu/Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2)**

![Model](https://img.shields.io/badge/model-Unlimited--OCR--2-2932E1)
![Quantization](https://img.shields.io/badge/quantization-INT8%20%7C%20NF4-orange)
![GPU](https://img.shields.io/badge/GPU-H200%20MIG%203g.71gb-76B900)
![PyTorch](https://img.shields.io/badge/torch-2.13.0-EE4C2C)
![Transformers](https://img.shields.io/badge/transformers-4.52.1-yellow)

</div>

---

## About

This repository measures how much **GPU memory**, **speed** and **OCR accuracy** change when
Baidu's **Unlimited-OCR-2** document OCR model is quantized with **bitsandbytes** (INT8 and NF4).

Unlimited-OCR-2 combines two vision encoders (SAM + CLIP-style) with a **DeepSeek-V2-style
Mixture-of-Experts** language model: 12 layers, 64 routed experts per layer (6 active per token),
about 3.2 B parameters in linear layers, 76.5% of them in the experts.

> [!NOTE]
> This is an independent benchmark project. It is not affiliated with or endorsed by Baidu.
> The logo belongs to the Unlimited-OCR-2 model repository.

## Key result

> [!TIP]
> **NF4 (4-bit) on all layers cuts peak GPU memory by 57% (7.39 → 3.17 GB) for ~17% slower
> inference, with the same OCR accuracy as bf16.**
> INT8 is not worth using: it saves less memory and is 2.5× slower.

## Results

17 test pages (3 real + 14 synthetic, all with ground truth), NVIDIA H200 MIG `3g.71gb`,
1 warm-up + 3 timed runs per page.

| | bf16 | INT8 | **NF4** | NF4 skip-vision |
|:--|--:|--:|--:|--:|
| **Peak GPU memory** | 7.39 GB | 4.88 GB | **3.17 GB** | 3.69 GB |
| vs bf16 | — | −34% | **−57%** | −50% |
| **Time per page** | **6.73 s** | 16.88 s | 7.86 s | 7.93 s |
| vs bf16 | — | 2.5× slower | **+17%** | +18% |
| Tokens / second | **38.0** | 15.1 | 32.5 | 32.2 |
| **Mean CER** | 0.069% | 0.075% | **0.068%** | 0.078% |
| Mean WER | 0.247% | 0.247% | **0.234%** | **0.234%** |

<details>
<summary><b>CER per page</b></summary>

| Page | bf16 | INT8 | NF4 | NF4 skip-vision |
|:--|--:|--:|--:|--:|
| Handwritten page | **0.30%** | 0.41% | 0.41% | 0.51% |
| Phone photo of printed page | 0.07% | 0.07% | **0.04%** | **0.04%** |
| Small scan (353×500) | 0.79% | 0.79% | **0.72%** | 0.79% |
| 14 synthetic pages (7 languages, forms, numbers, code, two columns, degraded, low-res) | 0.00% | 0.00% | 0.00% | 0.00% |

</details>

### Takeaways

- **NF4 is the best option**: about half the memory, small slowdown, same accuracy.
- **INT8 is slow**: bitsandbytes LLM.int8 splits every matrix multiply into int8 and fp16 parts.
- **Accuracy is equal within noise** for all variants: a few characters out of ~20,000.
- **Quantizing the vision encoders is safe** here: keeping them in bf16 costs +0.52 GB for no gain.
- **Quantization saves memory only through the weights**: ~1.1 GB of runtime overhead remains in every variant.

> [!IMPORTANT]
> Only 3 real pages were tested; the synthetic pages are read perfectly by every variant.
> Accuracy differences below ~0.1% CER are not significant. See [benchmark.md](benchmark.md#limitations).

## Variants

| Variant | Script | Quantized | Kept in bf16 |
|:--|:--|:--|:--|
| bf16 | `scripts/run_ocr.py` | nothing | everything |
| INT8 | `scripts/int8_quantize.py` | language model | SAM, vision encoder, projector, `lm_head` |
| NF4 | `scripts/int4_quantize.py` | all linear layers | none |
| NF4 skip-vision | `scripts/int4_quantize.py --skip-vision` | language model | SAM, vision encoder, projector, `lm_head` |

## Quick start

> [!WARNING]
> Use **torch 2.13.0**. Newer torch ships Triton 3.9, which breaks the model's custom attention kernels.

```bash
pip install -r requirements.txt
```

Run each variant on every image in `data/` (results go to `results/<variant>.json`):

```bash
python scripts/run_ocr.py
python scripts/int8_quantize.py
python scripts/int4_quantize.py
python scripts/int4_quantize.py --skip-vision
```

On a Slurm cluster:

```bash
sbatch run_ocr.sbatch
sbatch test_quant.sbatch scripts/int4_quantize.py
```

Each script accepts `--images` (paths or globs), `--output`, `--warmup` and `--runs`. Pages with a
matching `.txt` file next to the image are scored with CER/WER.

## Repository structure

```
.
├── benchmark.md            # full benchmark report
├── data/                   # test pages (.png) + ground truth (.txt)
├── docs/                   # working notes (model structure, results log, GPTQ/AWQ, KV cache)
├── results/                # benchmark output (JSON)
├── scripts/
│   ├── bench.py            # shared timing, memory and CER/WER measurement
│   ├── run_ocr.py          # bf16 baseline
│   ├── int8_quantize.py    # bitsandbytes INT8
│   ├── int4_quantize.py    # bitsandbytes NF4 (--skip-vision option)
│   ├── inspect_model.py    # layer inventory
│   └── make_test_pages.py  # synthetic test pages with exact ground truth
├── run_ocr.sbatch          # Slurm job: bf16
└── test_quant.sbatch       # Slurm job: quantized variants
```

## Documentation

| Document | Content |
|:--|:--|
| [benchmark.md](benchmark.md) | full report: setup, results, conclusions, limitations |
| [docs/note1.md](docs/note1.md) | layer inventory and model structure |
| [docs/note2.md](docs/note2.md) | benchmark log (1, 3 and 17 pages) |
| [docs/note3.md](docs/note3.md) | AWQ / GPTQ and why tools do not recognise this model |
| [docs/note4.md](docs/note4.md) | KV cache memory calculation |

## Acknowledgements

- [baidu/Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2) for the model
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) for INT8 / NF4 quantization
- [jiwer](https://github.com/jitsi/jiwer) for CER / WER
