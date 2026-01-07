# DeepSeek-OCR Finetune Dockerfile (clean, no options, no vLLM)
# CUDA 11.8 + Python 3.10 + Torch 2.6.0 + flash-attn + bitsandbytes

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_HOME=/usr/local/cuda \
    MAX_JOBS=8

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /deepseekOCR

RUN python -m pip install --upgrade pip setuptools wheel

# ---- PyTorch 2.6.0 (CUDA 11.8 wheels) ----
RUN pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# ---- HF transfer helper ----
RUN pip install hf_transfer

# ---- Install deps from requirements ----
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- bitsandbytes (4/8bit) ----
RUN pip install bitsandbytes==0.43.3

# ---- flash-attn ----
RUN pip install flash-attn==2.7.3 --no-build-isolation

CMD ["/bin/bash"]
