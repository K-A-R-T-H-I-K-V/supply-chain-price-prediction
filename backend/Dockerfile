FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ARG MODEL_DOWNLOAD_URL=""
RUN if [ ! -f ml/model.pkl ] && [ -n "$MODEL_DOWNLOAD_URL" ]; then \
      curl -fsSL -o ml/model.pkl "$MODEL_DOWNLOAD_URL"; \
    fi

RUN test -f ml/model.pkl && [ "$(wc -c < ml/model.pkl)" -gt 1000000 ] || ( \
      echo "ERROR: ml/model.pkl missing or LFS pointer not pulled" && exit 1 \
    )

ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg
EXPOSE 5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]