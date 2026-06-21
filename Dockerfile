# Use a slim Python 3.13 image for a lightweight runtime
FROM python:3.13-slim-bookworm

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for video processing and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
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
