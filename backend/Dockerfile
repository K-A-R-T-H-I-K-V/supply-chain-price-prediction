# Backend API Server
#
# Render settings (use these exactly):
#   Root Directory  = backend
#   Dockerfile Path = Dockerfile
#
# model.pkl is provided via Git LFS (git lfs track "backend/ml/*.pkl").
# Alternatives: MODEL_DOWNLOAD_URL build arg, or TRAIN_MODEL_AT_BUILD=true.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (includes ml/model.pkl via Git LFS and ml/train_model.py)
COPY . .

ARG MODEL_DOWNLOAD_URL=""
RUN if [ ! -f ml/model.pkl ] && [ -n "$MODEL_DOWNLOAD_URL" ]; then \
      echo "Downloading model from MODEL_DOWNLOAD_URL..." && \
      curl -fsSL -o ml/model.pkl "$MODEL_DOWNLOAD_URL"; \
    fi

ARG TRAIN_MODEL_AT_BUILD=false
RUN if [ ! -f ml/model.pkl ] && [ "$TRAIN_MODEL_AT_BUILD" = "true" ]; then \
      echo "Training model at build time..." && \
      MPLBACKEND=Agg python ml/train_model.py; \
    fi

RUN test -f ml/model.pkl && [ "$(wc -c < ml/model.pkl)" -gt 1000000 ] || ( \
      echo "ERROR: ml/model.pkl missing or is a Git LFS pointer (not the real file)." && \
      echo "Run: git lfs push --all origin" && \
      echo "Or set MODEL_DOWNLOAD_URL on Render." && \
      exit 1 \
    )

ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
