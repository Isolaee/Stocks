# ── Stage 1: base PyTorch image with CUDA support ────────────────────────────
# pytorch/pytorch images include conda; we use pip-only for a leaner result.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Avoid interactive apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app

# Copy requirements first so Docker can cache this layer
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY gcs.py         .
COPY preprocess.py  .
COPY model.py       .
COPY train.py       .

# ── Runtime defaults ──────────────────────────────────────────────────────────
# These are overridden by env vars passed to the Vertex AI custom job.
ENV PYTHONUNBUFFERED=1 \
    CHUNK_DIR=/tmp/data_chunks \
    CHECKPOINT_DIR=/tmp/checkpoints \
    GCS_OUTPUT_PREFIX=output \
    CHUNKED=1 \
    EPOCHS=100 \
    BATCH_SIZE=64 \
    LR=0.001 \
    PATIENCE=10 \
    RESUME=0

ENTRYPOINT ["python", "train.py"]
