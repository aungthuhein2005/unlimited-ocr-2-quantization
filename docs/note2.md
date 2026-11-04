# Quantization results: bf16 vs INT8 vs NF4

Follows [note1.md](note1.md) (layer inventory). This note records the first benchmark of
`baidu/Unlimited-OCR-2` in bf16, bitsandbytes INT8 and bitsandbytes NF4 (4-bit).

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA H200, MIG slice `3g.71gb` (same slice for every run) |
| Software | torch 2.13.0+cu130, Triton 3.7.1, transformers 4.52.1, bitsandbytes 0.50.2 |
| Test page | `data/en_image.png`: handwritten English page (EWD1200-0), 862×1024, 1,513 image tokens |
| Ground truth | `data/en_image.txt` (exact transcription, keeps the author's spelling "discription") |
| Measurement | `scripts/bench.py`: 2 warm-up runs, then 3 timed runs; median latency reported |
| Accuracy | CER/WER with `jiwer`, after stripping `<BLOCK>`/`<FORK>` tags and box coordinates |

### Variants

| Variant | Script | What is quantized |
|---|---|---|
| bf16 | `scripts/run_ocr.py` | nothing (baseline) |
| INT8 | `scripts/int8_quantize.py` | language model (incl. MoE experts); `sam_model`, `vision_model`, `projector`, `lm_head` kept in bf16 |
| NF4 | `scripts/int4_quantize.py` | **all** linear layers, incl. vision encoders (NF4 + double quant, bf16 compute) |

## Results (single page, jobs 6618–6620)

| | bf16 | INT8 | NF4 |
|---|---|---|---|
| Weights | 6.23 GB | 3.80 GB (−39%) | **2.04 GB (−67%)** |
| Peak memory | 7.39 GB | 4.88 GB | **3.17 GB** |
| Peak − weights | 1.16 GB | 1.08 GB | 1.13 GB |
| Median latency | **10.09 s** | 25.13 s (2.5× slower) | 11.61 s (+15%) |
| Tokens/s | **37.9** | 15.2 | 32.6 |
| Output tokens | 382 | 382 | 379 |
| Load time | 6.2 s | 8.1 s | 8.7 s |
| CER | **0.30%** | 0.41% | 0.41% |
| WER | 1.66% | 1.66% | 1.66% |
| Same output in all runs | yes | yes | yes |

GB = GiB (1024³ bytes).

### Errors vs ground truth

| Ground truth | bf16 | INT8 | NF4 |
|---|---|---|---|
| `EWD1200-0` (header) | `EvD1200-0` | `EV-D1200-0` | `EV-D1200-0` |
| `discription` (author's spelling) | `description` | `description` | `description` |
| `Volume 102 Number` | `102. Number` | `102. Number` | `102. Number` |

### Layout differences

- **INT8**: block types and boxes match bf16 almost exactly (one coordinate off by 1).
- **NF4**: the title lost its `title<LEVEL1>` label (output as plain `text`); boxes shift by a few units.

## Conclusions (so far)

1. **NF4 is the best trade-off on this page.** It cuts weights to a third (6.23 → 2.04 GB)
   and peak memory by 4.2 GB, for 15% lower speed and one extra character error.
2. **bitsandbytes INT8 is not worth it here.** It saves less memory than NF4 and is 2.5× slower,
   because LLM.int8 splits each matmul into int8 and fp16 parts.
3. **Quantization saves memory only through the weights.** Peak minus weights stays ~1.1 GB in
   all variants: that is image processing and text generation, which weight quantization
   does not shrink.
4. **Text accuracy is almost unchanged.** Both quantized variants add one character error in the
   small header box; all body text is identical to bf16. (Over 3 pages NF4 ties bf16: see
   "Accuracy on all 3 pages" below.)
5. **Layout is more sensitive than text.** NF4 dropped a block label, so CER alone does not show
   every regression. Check block types too if layout matters.
6. **The model sometimes corrects spelling** ("discription" → "description") in every variant.
   For strict transcription this counts as an error.
7. Measured weights match the layer inventory estimate: ~5.9 GiB of linear layers + ~0.3 GiB
   token embeddings ≈ 6.23 GiB.

## Problems found and fixed on the way

| Problem | Cause | Fix |
|---|---|---|
| `ImportError: torchvision` | model code imports it | added to `requirements.txt` |
| Triton `CompilationError: Block pointers have been removed` | torch 2.15 ships Triton 3.9, which removed `tl.make_block_ptr` used by the model's kernels | pinned `torch==2.13.0`, `torchvision==0.28.0` (Triton 3.7.1) |
| INT8 `ValueError: Only two or three dimensional matrices are supported` | SAM encoder feeds 4D tensors to its linears; bnb int8 matmul accepts 2D/3D only | `llm_int8_skip_modules` keeps vision side in bf16 |
| `invalid partition specified: gpu` | cluster has only `defq` with MIG slices | `--partition=defq --gres=gpu:3g.71gb:1` |

## Limitations

- **Few pages.** 3 pages, ~6,400 characters in total; CER differences of a few characters are not
  significant.
- **NF4 and INT8 are not like-for-like:** NF4 also quantizes the vision encoders, INT8 does not.
- **Ground truth is a manual transcription.** Uncertain spots: the dot after "102" (`en_image`) and
  `…` vs `...` (`en_image2`).
- **Margin line numbers are excluded** from `en_image2.txt`; every variant transcribes them, so they
  add the same ~16 character errors to all three.

## Multi-page benchmark (3 pages, jobs up to 6651)

All images in `data/` per variant, 1 warm-up + 3 timed runs per image, same `3g.71gb` slice:

| Image | Content | Size | Ground truth |
|---|---|---|---|
| `en_image.png` | handwritten page | 862×1024 | yes |
| `en_image1.png` | printed press release, phone photo | 960×1280 | yes |
| `en_image2.png` | printed text with margin line numbers, very small | 353×500 | yes (margin numbers excluded) |

### Summary (from `results/<variant>.json` → `summary`)

| | bf16 | INT8 | NF4 |
|---|---|---|---|
| Peak memory (max over pages) | 7.39 GB | 4.88 GB | **3.17 GB** |
| Mean latency per page | **15.81 s** | 39.53 s (2.5× slower) | 18.34 s (+16%) |
| Tokens/s (all pages) | **38.4** | 15.4 | 33.1 |
| Mean CER (3 pages) | **0.39%** | 0.42% | **0.39%** |
| Same output in all runs | yes | yes | yes |

The HPC jobs scored only `en_image.png` (the other ground truths were written afterwards). The CER
here is computed from the same saved predictions with the same tag stripping as `bench.py`.

### What the 3-page run adds

1. **The speed ranking holds across pages.** INT8 is 2.5× slower and NF4 ~15% slower than bf16, the
   same ratios as on one page. Tokens/s barely changes per variant, so speed depends on the variant,
   not on the page.
2. **Peak memory depends on the image size, not on how much text is written.** The two larger
   images (handwritten 862×1024 and photo 960×1280) reach the same peak; the small scan (353×500)
   needs ~0.6 GB less in every variant, even though it produces the most text.
3. **All variants are deterministic** on all pages: 3 runs gave identical text every time.
4. **Conclusions above still stand.** NF4 remains the best trade-off; INT8 remains the worst.
5. **Time grows with the amount of text.** The printed pages produce ~1.9× more tokens than the
   handwritten page and take ~1.9× longer, in every variant.
6. **NF4 output differs slightly from bf16 on the printed pages too** (707 vs 708 and 733 vs 732
   tokens), while INT8 matches bf16's token count on every page. The accuracy table below shows
   NF4's differences are not worse overall: one fix, one new misread.

### Accuracy on all 3 pages

| Page | bf16 | INT8 | NF4 |
|---|---|---|---|
| `en_image` (handwritten) | **0.30%** | 0.41% | 0.41% |
| `en_image1` (photo) | 0.07% | 0.07% | **0.04%** |
| `en_image2` (small scan) | 0.79% | 0.79% | **0.72%** |
| **Mean CER** | **0.39%** | 0.42% | **0.39%** |

| Ground truth | bf16 | INT8 | NF4 |
|---|---|---|---|
| `sheets", commented` (`en_image1`) | `sheets”` | `sheets”` | `sheets”` |
| `Kazakh Cyrillic` (`en_image1`) | `Kazakh, Cyrillic` | `Kazakh, Cyrillic` | **correct** |
| `Saffron Hill -the` (`en_image2`) | `Hill - the` | `Hill - the` | `Hill - the` |
| `newly made …` (`en_image2`) | `...` | `...` | `...` |
| `obscene` (`en_image2`, after margin "5") | `5obscene` | `5obscene` | `50bscene` (o → 0) |
| margin numbers 10–35 (`en_image2`) | included | included | included |

7. **NF4 ties bf16 on accuracy over 3 pages.** It is worse on the handwritten page and better on
   the two printed pages. The single-page result (bf16 best) was misleading.
8. **INT8 makes exactly bf16's errors** on both printed pages, matching its output staying closest
   to bf16.
9. **All three are within ~0.03% CER.** On 3 pages that is too small to rank accuracy. Memory and
   speed are where the variants clearly differ.

### Per page

| Image | bf16 | INT8 | NF4 |
|---|---|---|---|
| `en_image.png` (handwritten, 862×1024) | 10.0 s · 382 tok · 7.39 GB | 25.4 s · 382 tok · 4.88 GB | 11.7 s · 379 tok · 3.17 GB |
| `en_image1.png` (photo, 960×1280) | 18.5 s · 708 tok · 7.39 GB | 46.4 s · 708 tok · 4.88 GB | 21.4 s · 707 tok · 3.17 GB |
| `en_image2.png` (small scan, 353×500) | 18.9 s · 732 tok · 6.79 GB | 46.8 s · 732 tok · 4.29 GB | 21.9 s · 733 tok · 2.57 GB |

Time = median of 3 runs; tok = output tokens; GB = peak allocated GPU memory.

## Next steps

1. Rerun the three jobs so the JSON files contain the 3-page accuracy.
2. Run NF4 with the same skip list as INT8, to separate the effect of quantizing the vision encoders.
3. Try GPTQ (see [note3.md](note3.md)).
4. Add more pages (printed scans, tables, other languages) before drawing firm conclusions.
