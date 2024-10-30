# Use a base image with CUDA support and PyTorch preinstalled
#FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu121.2-2.py310

# Set environment variables for Python and CUDA
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_DEVICE_ORDER=PCI_BUS_ID

# Install essential packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --upgrade pip

# Install PyTorch with CUDA and transformers library
RUN pip3 install torch torchvision torchaudio 
# --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install transformers

# Set the working directory
WORKDIR /app

# Copy your script into the container
COPY scripts/run_model.py /app/

# Ensure the container runs the Python script on startup
CMD ["python3", "run_model.py"]
