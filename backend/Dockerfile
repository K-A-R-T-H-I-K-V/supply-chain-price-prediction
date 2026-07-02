# Backend API Server
# Render settings: Root Directory = (empty), Dockerfile Path = backend/Dockerfile
# If Root Directory is set to "backend", repo-root paths like data/ won't be in the build context.
#
# model.pkl is gitignored — provide it one of these ways:
#   A) Git LFS (recommended): git lfs track backend/ml/*.pkl && git push
#   B) Release download: set Render Docker build arg MODEL_DOWNLOAD_URL to your
#      release asset URL (upload with: gh release upload TAG backend/ml/model.pkl)
#   C) Build-time training: set TRAIN_MODEL_AT_BUILD=true (slow; may timeout on free tier)
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (includes ml/model.pkl via Git LFS and ml/train_model.py)
COPY backend/ .

# Training data only needed when TRAIN_MODEL_AT_BUILD=true (requires repo-root build context)
ARG TRAIN_MODEL_AT_BUILD=false


# Download pre-trained model from a GitHub release (or other HTTPS URL)
ARG MODEL_DOWNLOAD_URL=""
RUN if [ ! -f ml/model.pkl ] && [ -n "$MODEL_DOWNLOAD_URL" ]; then \
      echo "Downloading model from MODEL_DOWNLOAD_URL..." && \
      curl -fsSL -o ml/model.pkl "$MODEL_DOWNLOAD_URL"; \
    fi

# Optional: train during image build (10–30+ min; needs matplotlib/seaborn)
ARG TRAIN_MODEL_AT_BUILD=false
RUN if [ ! -f ml/model.pkl ] && [ "$TRAIN_MODEL_AT_BUILD" = "true" ]; then \
      echo "Training model at build time..." && \
      MPLBACKEND=Agg python ml/train_model.py; \
    fi

# Reject Git LFS pointer files (~130 bytes) — Render must pull LFS during clone
RUN test -f ml/model.pkl && [ "$(wc -c < ml/model.pkl)" -gt 1000000 ] || ( \
      echo "ERROR: ml/model.pkl missing or is a Git LFS pointer (not the real 162MB file)." && \
      echo "Push with: git lfs push --all origin" && \
      echo "Or set MODEL_DOWNLOAD_URL / TRAIN_MODEL_AT_BUILD=true on Render." && \
      exit 1 \
    )

ENV PYTHONUNBUFFERED=1

ENV MPLBACKEND=Agg

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
