According to the layer inventory result

there are total 2,390 linear layer with three fields()

---

# Layer inventory analysis: baidu/Unlimited-OCR-2

Source: `results/layer_inventory.json`, produced by `scripts/inspect_model.py` (HPC job, CPU, bf16).

## Summary

- 2,390 `nn.Linear` layers, about **3.17B parameters** in linear layers.
- Three parts: image encoders (SAM + CLIP-style, ~12%), a 1-layer projector (0.1%), and a 12-layer **Mixture-of-Experts** language model.
- **76.5%** of linear parameters are in the MoE experts, so they are the main quantization target.
- Quantizing only the experts to 4-bit: linear weights go from ~6.3 GB to ~2.7 GB (estimate).

## Step 1: Read the file

Each entry is one linear layer with three fields:

```json
{ "name": "model.layers.3.mlp.experts.17.up_proj", "in_features": 1280, "out_features": 896 }
```

Counting the entries gives **2,390 linear layers**.

## Step 2: Collapse numbers in names to find patterns

Replacing every number in a name with `N` and counting turns 2,390 names into 28 patterns, e.g.:

```
704  model.layers.N.mlp.experts.N.up_proj            1280x896
 24  model.vision_model.transformer.layers.N.mlp.fc1  1024x4096
 12  model.sam_model.blocks.N.mlp.lin1                768x3072
  1  lm_head                                          1280x129280
```

## Step 3: Read the structure from names and counts

| Part | Evidence |
|---|---|
| `sam_model` | 12 blocks, width 768 (Segment Anything encoder) |
| `vision_model` | 24 layers, width 1024, `qkv_proj`/`fc1`/`fc2` (CLIP-style ViT) |
| `projector` | 1 layer, 2048 → 1280 (two 1024-wide image features → LM width 1280) |
| `model.layers.N` | language model, 12 layers (0–11) |

Why it is Mixture-of-Experts:

- 704 `experts.N.up_proj` over 11 layers (1–11): 704 ÷ 11 = **64 experts per layer**.
- Each of layers 1–11 has 203 linears: 4 attention (q/k/v/o) + 4 small router projections (`indexer`, `dash_router` q/k) + 64×3 experts + 3 shared expert = **203**.
- Layer 0 has 11: 4 + 4 + 3 dense MLP (`mlp.gate_proj` 1280×6848). So **layer 0 is dense**, layers 1–11 are MoE.

## Step 4: Count parameters

Weight size of a linear layer = `in_features × out_features`.

| Group | Calculation | Parameters | Share |
|---|---|---|---|
| MoE experts | 704 × 3 × (1280 × 896) | 2,422.2M | 76.5% |
| Vision (CLIP-style) | 24 × (2×1024×4096 + 1024² + 1024×3072) | 302.0M | 9.5% |
| `lm_head` | 1280 × 129,280 | 165.5M | 5.2% |
| Attention (incl. indexer/router) | | 86.5M | 2.7% |
| SAM | | 84.9M | 2.7% |
| Shared experts | | 75.7M | 2.4% |
| Layer 0 dense MLP | | 26.3M | 0.8% |
| Projector | 2048 × 1280 | 2.6M | 0.1% |
| **Total** | | **3,166M** | |

## Step 5: Estimate memory

- bf16 = 2 bytes/param: 3.166B × 2 = **6.33 GB**
- Experts in bf16: 2.422B × 2 = 4.84 GB → in 4-bit (~0.5 bytes/param): **1.21 GB**
- Rest stays bf16: 0.744B × 2 = 1.49 GB
- After quantizing experts: 1.21 + 1.49 ≈ **2.70 GB** (saves ~3.6 GB)

## Confidence

**Measured directly from the file**
- 2,390 layers, ~3.17B linear params, 76.5% expert share
- 64 experts/layer, dense layer 0, layer shapes

**Calculated (approximate)**
- Memory numbers. They exclude 4-bit scale overhead and non-Linear weights (token embeddings, norms, SAM patch embedding, MoE router), so the full model is somewhat larger than 3.17B. The router is not an `nn.Linear` (it does not appear in the file).

**Interpretation, not yet verified**
- "Similar to DeepSeek-OCR": inferred from the SAM + CLIP + MoE layout and layer names.
- "Only a few experts run per token": general MoE behaviour; check `config.json` for the actual number of active experts.
- "Keep vision encoders, projector and `lm_head` in bf16": common practice for vision-language models. **Hypothesis to test** by comparing OCR accuracy with and without quantization.

## Next step

Load the model with 4-bit quantization of the expert layers only (keep vision, SAM, projector, `lm_head`, indexer/router in bf16). Run OCR on test images and compare accuracy (`jiwer` CER/WER), memory and speed against the bf16 baseline.

