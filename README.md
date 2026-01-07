# deepseekOCR
DeepSeekOCR Korean fine-tuning for structured document OCR

```
docker compose up -d --build
```

# 컨테이너 내부 코드 수정
```
python - <<'PY'
from pathlib import Path
p = Path("/root/.cache/huggingface/modules/transformers_modules/deepseek-ai/DeepSeek-OCR/9f30c71f441d010e5429c532364a86705536c53a/modeling_deepseekocr.py")
lines = p.read_text().splitlines(True)

# 1-indexed line numbers
start, end = 505, 507

# 505줄의 선행 공백(들여쓰기)을 그대로 가져와서 3줄 모두에 동일 적용
indent = lines[start-1][:len(lines[start-1]) - len(lines[start-1].lstrip())]

new_block = [
    f"{indent}mask = images_seq_mask[idx].unsqueeze(-1).to(device=inputs_embeds[idx].device)\n",
    f"{indent}src  = images_in_this_batch.to(device=inputs_embeds[idx].device, dtype=inputs_embeds[idx].dtype)\n",
    f"{indent}inputs_embeds[idx].masked_scatter_(mask, src)\n",
]

lines[start-1:end] = new_block
p.write_text(''.join(lines))
print("Replaced lines 505-507 with correct indentation.")
PY

```
