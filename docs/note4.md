# KV cache memory for Unlimited-OCR-2

Follows [note2.md](note2.md) (benchmark results). This note calculates how much GPU memory the
KV cache needs and compares it with what we measured.

## Summary

- The KV cache needs **60 KiB per token** in bf16.
- For one handwritten page (1,901 tokens) that is **~0.11 GiB**, under 10% of the measured
  memory overhead. Most of the overhead is image processing, not the KV cache.
- At the maximum context (131,072 tokens) it would be **7.5 GiB**, more than the bf16 weights.
- Weight quantization (INT8, NF4) **does not shrink** the KV cache; it stays bf16 in every variant.

## Formula

$$
M_{\mathrm{KV}} = 2 \times L \times H_{KV} \times D \times S \times B \times P
$$

## Values

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| **2** | one Key + one Value tensor | 2 | formula |
| **L** | layers | **12** | `num_hidden_layers` |
| **H_KV** | key/value heads | **10** | `num_key_value_heads` (= `num_attention_heads`: normal multi-head attention, no GQA) |
| **D** | size per head | **128** | `hidden_size / heads` = 1280 / 10; `head_dim=128` in the job log |
| **S** | sequence length (tokens) | **1,901** | prompt 1,519 (`S=1519` in log: 1,513 image tokens + 6) + 382 output tokens (`en_image`) |
| **B** | batch size | **1** | one page per request |
| **P** | bytes per value | **2** | bf16 |

`config.json` has `use_mla: false` and `kv_lora_rank: null`, so the KV cache is **not compressed**
(unlike the original DeepSeek-V2 MLA). The standard formula applies directly.

## Calculation

### Step 1: per token

$$
2 \times 12 \times 10 \times 128 \times 2 = 61{,}440 \text{ bytes} = \mathbf{60\ KiB\ per\ token}
$$

### Step 2: handwritten page (S = 1,901, B = 1)

$$
61{,}440 \times 1{,}901 = 116{,}797{,}440 \text{ bytes} \approx \mathbf{111\ MiB} \approx \mathbf{0.11\ GiB}
$$

### Step 3: maximum context (S = 131,072, B = 1)

$$
61{,}440 \times 131{,}072 = 8{,}053{,}063{,}680 \text{ bytes} = \mathbf{7.5\ GiB}
$$

### Other cases

| Case | S | B | KV cache |
|---|---|---|---|
| Handwritten page | 1,901 | 1 | 0.11 GiB |
| Same page, batch of 16 | 1,901 | 16 | 1.74 GiB |
| Maximum context | 131,072 | 1 | 7.5 GiB |

## Comparison with measurements

| `en_image.png` | bf16 | INT8 | NF4 |
|---|---|---|---|
| Weights | 6.23 GB | 3.80 GB | 2.04 GB |
| Peak memory | 7.39 GB | 4.88 GB | 3.17 GB |
| **Peak − weights** (overhead) | 1.16 GB | 1.08 GB | 1.13 GB |
| KV cache (calculated) | 0.11 GiB | 0.11 GiB | 0.11 GiB |
| KV share of overhead | ~9% | ~10% | ~10% |

(GB here = GiB, as in note2.)

## What this tells us

1. **For one page the KV cache is small.** It is ~0.11 GiB of the ~1.1 GB overhead. The rest (~1 GB)
   is mostly **working memory**: image processing in the vision encoders (SAM + CLIP-style) and
   temporary buffers during generation.
2. **Weight quantization does not reduce the KV cache.** INT8 and NF4 shrink only the weights; the
   cache stays bf16. This is why peak − weights is ~1.1 GB in all three variants.
3. **The KV cache matters when S or B grows:** multi-page documents (the model supports several
   pages per prompt) or batching many pages. Then **KV cache quantization** (FP8/INT8 KV) would
   halve it and becomes worth testing.
4. **The overhead also changes with image size.** The small scan (`en_image2`, 353×500) had ~0.6 GB
   lower peak than the larger images (note2), even with more output tokens. Image size affects
   memory more than text length at B = 1.

## Caveat

The custom DASH attention has an extra `indexer.k_proj` (1280 → 128) in each layer. If its keys are
also cached, that adds 12 × 128 × 2 = **3 KiB per token (+5%)**: still negligible. Not yet checked in
the model code.

## Next steps

- Check whether the DASH indexer keys are cached (read `dash_attention.py`).
- If multi-page or batched inference is planned, measure the KV cache directly (peak memory vs S)
  and test FP8/INT8 KV cache quantization.
