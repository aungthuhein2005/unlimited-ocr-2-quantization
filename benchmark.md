# Unlimited-OCR-2 Quantization Benchmark

Final results of quantizing [`baidu/Unlimited-OCR-2`](https://huggingface.co/baidu/Unlimited-OCR-2) with
bitsandbytes, measured on an HPC GPU. Detailed working notes are in [`docs/`](docs/).

## Result in one line

**NF4 (4-bit) on all layers cuts peak GPU memory by 57% (7.39 → 3.17 GB) for ~17% slower inference,
with the same OCR accuracy as bf16. INT8 is not worth using: it saves less memory and is 2.5× slower.**

The NF4 model is published as
[aungthuhein-dev/Unlimited-OCR-2-bnb-nf4](https://huggingface.co/aungthuhein-dev/Unlimited-OCR-2-bnb-nf4)
(load with `torch_dtype=torch.bfloat16`).

## Model

| | |
|---|---|
| Model | `baidu/Unlimited-OCR-2` (`model_type: unlimited-ocr`, custom code) |
| Vision | SAM encoder (12 blocks) + CLIP-style encoder (24 layers) + 1-layer projector |
| Language model | DeepSeek-V2-style MoE, 12 layers, hidden size 1280, MHA (10 heads × 128) |
| MoE | layer 0 dense; layers 1–11: 64 routed experts (6 active per token) + 2 shared experts |
| Linear layers | 2,390 layers, ~3.17 B parameters (76.5% in routed experts) |

Details: [`docs/note1.md`](docs/note1.md).

## Setup

| | |
|---|---|
| GPU | NVIDIA H200, MIG slice `3g.71gb` (same slice for every run) |
| Software | torch 2.13.0+cu130, Triton 3.7.1, transformers 4.52.1, bitsandbytes 0.50.2 |
| Procedure | per page: 1 warm-up run, then 3 timed runs; median latency reported |
| Accuracy | CER / WER with `jiwer`, after removing layout tags and box coordinates |
| Code | `scripts/bench.py` (shared measurement), `scripts/run_ocr.py`, `scripts/int8_quantize.py`, `scripts/int4_quantize.py` |

## Variants

| Variant | Method | Quantized | Kept in bf16 |
|---|---|---|---|
| **bf16** | none (baseline) | nothing | everything |
| **INT8** | bitsandbytes LLM.int8 | language model (incl. experts) | `sam_model`, `vision_model`, `projector`, `lm_head` |
| **NF4** | bitsandbytes NF4 + double quant, bf16 compute | all linear layers | none |
| **NF4 skip-vision** | same as NF4 | language model (incl. experts) | `sam_model`, `vision_model`, `projector`, `lm_head` |

All four are data-free (no calibration). GPTQ / AWQ were not tested.

## Test set (17 pages, all with ground truth)

| Type | Pages | What it tests |
|---|---|---|
| **Real** | `en_image` (handwritten), `en_image1` (phone photo of printed page), `en_image2` (small 353×500 scan with margin line numbers) | real-world quality |
| **Synthetic** | 14 pages from `scripts/make_test_pages.py` | report, two columns, form, invoice numbers, degraded copy, tiny image, code; German, French, Spanish, Russian, Chinese, Japanese, Korean |

Synthetic pages are rendered from known text, so their ground truth is exact. Real-page ground truth
is a manual transcription (margin line numbers excluded).

## Results

### Summary (17 pages)

| | bf16 | INT8 | NF4 | NF4 skip-vision |
|---|---|---|---|---|
| Weights | 6.23 GB | 3.80 GB | **2.04 GB** | 2.58 GB |
| **Peak GPU memory** | 7.39 GB | 4.88 GB | **3.17 GB** | 3.69 GB |
| Peak memory vs bf16 | — | −34% | **−57%** | −50% |
| **Mean time per page** | **6.73 s** | 16.88 s | 7.86 s | 7.93 s |
| Time vs bf16 | — | +151% (2.5×) | **+17%** | +18% |
| Tokens / second | **38.0** | 15.1 | 32.5 | 32.2 |
| **Mean CER** | 0.069% | 0.075% | **0.068%** | 0.078% |
| Mean WER | 0.247% | 0.247% | **0.234%** | **0.234%** |
| Same output in all 3 runs | yes | yes | yes | yes |

GB = GiB (1024³ bytes). Weights from the model's memory footprint after loading.

### CER per page

| Page | bf16 | INT8 | NF4 | NF4 skip-vision |
|---|---|---|---|---|
| `en_image` (handwritten) | **0.30%** | 0.41% | 0.41% | 0.51% |
| `en_image1` (photo) | 0.07% | 0.07% | **0.04%** | **0.04%** |
| `en_image2` (small scan) | 0.79% | 0.79% | **0.72%** | 0.79% |
| 14 synthetic pages | 0.00% | 0.00% | 0.00% | 0.00% |

### Speed and memory per real page (3-page run)

| Page | bf16 | INT8 | NF4 |
|---|---|---|---|
| `en_image` (862×1024) | 10.0 s · 382 tok · 7.39 GB | 25.4 s · 382 tok · 4.88 GB | 11.7 s · 379 tok · 3.17 GB |
| `en_image1` (960×1280) | 18.5 s · 708 tok · 7.39 GB | 46.4 s · 708 tok · 4.88 GB | 21.4 s · 707 tok · 3.17 GB |
| `en_image2` (353×500) | 18.9 s · 732 tok · 6.79 GB | 46.8 s · 732 tok · 4.29 GB | 21.9 s · 733 tok · 2.57 GB |

## Conclusions

1. **NF4 on all layers is the best option.** 57% less peak memory, 17% slower, same accuracy as bf16.
2. **INT8 is not worth using.** It saves less memory than NF4 and is 2.5× slower, because
   bitsandbytes LLM.int8 splits every matrix multiply into int8 and fp16 parts.
3. **Accuracy is the same for all variants within noise.** The differences are a few characters out
   of ~20,000. Memory and speed are the real differences.
4. **Quantizing the vision encoders does no measurable harm.** Keeping them in bf16 (NF4 skip-vision)
   costs +0.52 GB with no gain in speed or accuracy.
5. **No variant breaks** on two-column layouts, forms, numbers, code, degraded or tiny images, or six
   non-English languages including Chinese, Japanese and Korean.
6. **Quantization saves memory only through the weights.** Peak minus weights stays ~1.1 GB in every
   variant: image processing and generation buffers, which weight quantization does not shrink.
7. **Speed depends on the variant, memory on the image.** Time grows with output length; peak memory
   grows with image size (the 353×500 scan needs ~0.6 GB less than the larger pages).
8. **Quantization makes the model fit on small GPUs.** bf16 needs ~7.4 GB; NF4 needs ~3.2 GB, which
   fits a 6 GB laptop GPU (not tested here).

## KV cache

60 KiB per token in bf16 (2 × 12 layers × 10 heads × 128 × 2 bytes). One handwritten page
(1,901 tokens) needs ~0.11 GiB, under 10% of the ~1.1 GB runtime overhead. At the maximum context
(131,072 tokens) it would be 7.5 GiB. Weight quantization does not reduce it. Details:
[`docs/note4.md`](docs/note4.md).

## Limitations

- **Small real test set.** Only 3 real pages; the 14 synthetic pages are read perfectly by every
  variant, so they cannot separate them. Accuracy rankings within ~0.1% CER are not significant.
- **Manual ground truth** for the real pages; two spots are uncertain (dot after "102" in
  `en_image`, `…` vs `...` in `en_image2`).
- **One GPU configuration** (H200 MIG `3g.71gb`) and **one page per request**; batching or other GPUs
  may change the speed ranking.
- **Data-free methods only.** No calibration-based quantization (GPTQ, AWQ) or mixed 4/8-bit
  precision was tested.
- **Layout not scored.** CER measures text only; NF4 once dropped a title label on the handwritten
  page.
- **Tables not tested.**

## Not covered (possible future work)

- GPTQ / AWQ with calibration: [`docs/note3.md`](docs/note3.md)
- Mixed precision (experts 4-bit, attention and shared experts 8-bit or bf16)
- Community builds: GGUF (patched llama.cpp), LiteRT-LM int8 (phones), MLX (Apple)
- On-device runs (laptop GPU, phone)

## Problems fixed during the project

| Problem | Cause | Fix |
|---|---|---|
| `ImportError: torchvision` | model code imports it | added to `requirements.txt` |
| Triton `CompilationError: Block pointers have been removed` | torch 2.15 ships Triton 3.9, which removed `tl.make_block_ptr` used by the model | pinned `torch==2.13.0`, `torchvision==0.28.0` |
| INT8 `ValueError: Only two or three dimensional matrices are supported` | SAM encoder passes 4D tensors; bnb int8 matmul accepts 2D/3D only | keep vision modules in bf16 for INT8 |
| `invalid partition specified: gpu` | cluster has only `defq` with MIG slices | `--partition=defq --gres=gpu:3g.71gb:1` |
| Exported NF4 crashes: `Input type (c10::BFloat16) and bias type (c10::Half)` | bnb models load non-quantized layers in float16 unless told otherwise | load with `torch_dtype=torch.bfloat16` |
| Exported NF4 outputs `0,0,0,...` until the token limit | `save_pretrained` wrote changed DASH settings to `config.json` (`dash_topk_blocks` 12 → 64, `dash_stage0` head_cls → indexer) | export keeps the original `config.json` plus `quantization_config` |

## Reproduce

On the HPC, from the project root:

```bash
sbatch run_ocr.sbatch
sbatch test_quant.sbatch scripts/int8_quantize.py
sbatch test_quant.sbatch scripts/int4_quantize.py
sbatch test_quant.sbatch scripts/int4_quantize.py --skip-vision
```

Results are written to `results/bf16.json`, `results/int8.json`, `results/int4.json` and
`results/int4_skipvision.json`. Regenerate the synthetic pages with `python scripts/make_test_pages.py`
(Windows fonts required).

## Notes

| Note | Topic |
|---|---|
| [`docs/note1.md`](docs/note1.md) | layer inventory and model structure |
| [`docs/note2.md`](docs/note2.md) | full benchmark log (1, 3 and 17 pages) |
| [`docs/note3.md`](docs/note3.md) | AWQ / GPTQ and why tools do not recognise this model |
| [`docs/note4.md`](docs/note4.md) | KV cache memory |
