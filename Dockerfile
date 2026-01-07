# DeepSeek-OCR Finetune Dockerfile
# CUDA 11.8 + Python 3.10+ 베이스 이미지
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Python 심볼릭 링크 설정
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

# 작업 디렉토리 설정
WORKDIR /deepseekOCR

# pip 업그레이드
RUN python -m pip install --upgrade pip setuptools wheel

# ---------- Core GPU stack (CUDA 11.8) ----------
RUN pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# ---------- Core libs ----------
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- Optional: flash-attn (non-fatal) ----------
RUN pip install flash-attn --no-build-isolation || \
    echo "[WARN] flash-attn installation failed. The training script will automatically use 'eager' attention."

# ---------- Optional: bitsandbytes (4/8bit) ----------
RUN pip install bitsandbytes==0.43.3 || \
    echo "[WARN] bitsandbytes installation failed. Quantization features will not be available."

# 프로젝트 파일은 볼륨 마운트로 연결 (COPY 제거)
# 코드 변경사항이 자동으로 반영됩니다

# Sanity check
RUN python -c "import torch, transformers, peft; \
    print('[INFO] torch:', torch.__version__, '| cuda available:', torch.cuda.is_available()); \
    print('[INFO] transformers:', transformers.__version__); \
    print('[INFO] peft:', peft.__version__)"

# 기본 명령어
CMD ["/bin/bash"]

