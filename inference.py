# vram: 8gb


import os
# ✅ GPU 하나만 강제 사용 (0번 카드)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.float16,
    device_map={"": 0},              
    _attn_implementation="flash_attention_2",
)

model.eval()

prompt = "<image>\nFree OCR"

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file="your_img.png",
    output_path="./output",
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=True,
)

print(res)
