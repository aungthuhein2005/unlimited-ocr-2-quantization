# AWQ and GPTQ for Unlimited-OCR-2

Follows [note1.md](note1.md) (layer inventory) and [note2.md](note2.md) (bitsandbytes results).

## Summary

- So far we used only **bitsandbytes** (INT8 and NF4): quantization **without calibration data**.
- **AWQ and GPTQ** are calibration-based methods. They usually keep more accuracy at 4-bit and
  can run faster than bitsandbytes.
- **The methods work on this model. The tools do not recognise it automatically**, because it is a
  new model type. We have to tell the tool how the model is built.

## Method vs tool

| | What it is | Example |
|---|---|---|
| **Method** | the technique for making weights smaller | GPTQ, AWQ |
| **Tool** | software that applies the method automatically | GPTQModel, llm-compressor |

**The method works on this model. The tool does not know the model**, so it cannot do the job by itself.

## What we used so far: bitsandbytes (no calibration)

| | How it works |
|---|---|
| **INT8 (LLM.int8)** | rounds each weight row to 8-bit at load time; rare large input values are computed in 16-bit (this split makes it slow: 2.5× slower in note2) |
| **NF4** | splits weights into blocks of 64, scales each block, rounds to the nearest of 16 fixed 4-bit values |

Both round each weight **on its own, without looking at real data**. Easy to use, but the simplest
baseline. The results in note2 are a **data-free baseline**.

## What GPTQ and AWQ do differently

Both run a small set of real inputs (**calibration data**) through the model first, then use what
they learn to quantize more carefully.

| | Idea | Result |
|---|---|---|
| **GPTQ** | quantizes one column at a time and adjusts the remaining weights to cancel the rounding error | usually more accurate than plain rounding at 4-bit |
| **AWQ** | finds the weights that matter most (from calibration inputs) and scales them up before quantizing, so they lose less precision | similar accuracy, often more robust |

Both save **already-packed 4-bit weights** and use fast GPU kernels, so they are often **faster**
than bitsandbytes NF4.

## Why the method works on this model

Every model is built from **linear layers** (tables of numbers). GPTQ and AWQ make each table
smaller (16-bit → 4-bit) while losing as little accuracy as possible.

This model has **2,390 linear layers** (note1), the same kind every model uses. So the method applies.

## Why the tool does not recognise it

A tool keeps a list of models it knows, with a "recipe card" for each:

- which layers to quantize and which to leave alone
- in what order
- how to feed calibration data through the model
- (AWQ only) which normalisation layer comes before each linear layer, to fold scales into

The tool finds the card by the model name in `config.json`:

```json
"model_type": "unlimited-ocr",
"architectures": ["UnlimitedOCRForCausalLM"]
```

No tool has a card for `"unlimited-ocr"`, so it does not know:

- to **skip the image parts** (`sam_model`, `vision_model`, `projector`)
- that the input is **an image**, not plain text (calibration must go through `infer()`)
- where the layers are and what they are called

## Good news: the language model is close to a known one

The inner language config says `"architectures": ["DeepseekOCRForCausalLM"]`, and the code file is
`modeling_deepseekv2.py`. So the text part is **DeepSeek-V2-style MoE**, which some tools already support.

It has custom additions, so a DeepSeek-V2 card may not match exactly:

- DASH attention with extra `indexer` and `dash_router` projections
- page-aware position encoding (`PageRope`)

### Model facts from `config.json` (confirms note1)

| Setting | Value |
|---|---|
| Routed experts per layer | 64 |
| Experts used per token | **6** |
| Shared experts | 2 (why `shared_experts` is 1792 = 2 × 896 wide) |
| Dense (non-MoE) first layers | 1 (layer 0) |
| Decoder layers | 12 |
| Hidden size | 1280 |

## Options

| Option | Effort | Notes |
|---|---|---|
| **A. Register a custom model definition** in a tool (map the layer names) | medium | the usual way to add an unsupported model; test with the installed tool version |
| **B. Write a small GPTQ loop ourselves** | medium | ~100 lines: capture each layer's inputs with hooks while running OCR on calibration pages, quantize, move on. Full control over skipping vision and covering experts. |
| **C. AWQ** | harder | needs the norm → linear mapping and scale folding for this exact architecture; easier after GPTQ works |
| (optional) **HQQ** | low | another no-calibration 4-bit method, often more accurate than NF4; easy extra baseline |

**Plan: GPTQ first** (option A, fall back to B if the tool fights the custom code), then AWQ.

## Plan for GPTQ

1. Quantize the **text part** (attention, MoE experts, shared experts, layer 0 MLP).
2. Keep in bf16: `sam_model`, `vision_model`, `projector`, `lm_head`, `indexer`, `dash_router`.
3. Calibrate with **50–100 real document pages** run through `infer()`.
4. Compare with the same pages and metrics as note2 (weights, peak memory, latency, tokens/s, CER/WER).

## Watch out: MoE calibration

Each token uses only **6 of 64 experts**. With too few calibration pages, some experts barely see
any data and get quantized badly.

- Use **enough, varied pages** (printed, handwritten, tables, different layouts).
- **Count calibration tokens per expert** and check that none is close to zero.

## Things to check before starting

- Which tool version supports custom MoE models best at install time (tool support changes quickly).
- That the packed 4-bit kernels run alongside the model's own Triton attention kernels
  (they only touch linear layers, so this should be fine, but test it).

## In one sentence

**GPTQ and AWQ can shrink this model; we just have to tell the tool how the model is built,
because it is a new model the tool has never seen.**
