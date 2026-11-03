
# scripts/int4_quantize.py
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)

MODEL_ID = "baidu/Unlimited-OCR-2"

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=True
)

model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
).eval()

print("Loaded NF4 candidate")

prediction = model.infer(
    tokenizer,
    image_file="data/image.png",
    output_path="results/int4",
    save_results=False,
)

print(prediction)
