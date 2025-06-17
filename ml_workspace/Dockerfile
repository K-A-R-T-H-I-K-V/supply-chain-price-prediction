FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# ✅ Install dependencies including libgomp1 for LightGBM
RUN apt-get update && apt-get install -y libgomp1 \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "scripts/train_model.py"]
