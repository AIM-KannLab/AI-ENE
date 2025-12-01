FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Basic env for reproducibility and smaller logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONHASHSEED=42 \
    SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

WORKDIR /app

# System deps and Python (include libGL for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/* \
  && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
  && python -m pip install --upgrade pip

# Install core DL stacks first so downstream deps resolve cleanly
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install "tensorflow==2.13.*" "keras==2.13.*"

# Then install the rest of the Python deps (nnU-Net, medpy, etc.)
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Install local utils used by AI-ENE
COPY ENE_inference/data-utils /app/ENE_inference/data-utils
RUN pip install -e /app/ENE_inference/data-utils

# Copy the rest of the project
COPY . /app

# Default: run the end-to-end pipeline; device decided by --gpu and docker --gpus
ENTRYPOINT ["python", "/app/run_e2e.py"]
CMD ["--gpu", "0"]


