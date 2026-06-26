# NVIDIA CUDA runtime with cuDNN for GPU-accelerated ML workloads
# (faster-whisper / CTranslate2 uses CUDA directly for transcription)
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Make all NVIDIA GPUs visible inside the container
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python 3.12, ffmpeg, and system dependencies required for
# video processing, OpenCV, and general ML workloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Copy the uv package manager binary into the image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Enable bytecode compilation for faster startup/execution
ENV UV_COMPILE_BYTECODE=1

# Copy package definition files to restore dependencies
COPY pyproject.toml uv.lock ./

# Install dependencies using uv with cache mount for speed
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Copy the application source code
COPY . .

# Run final sync to install the project package itself if necessary
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Place the virtual environment's executables on the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose the default FastAPI port
EXPOSE 8000

# Start the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
