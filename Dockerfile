# Backend image for Hugging Face Spaces (Docker SDK).
# HF Spaces expects the app to listen on port 7860.
FROM python:3.12-slim

# libgomp1 is required by xgboost's native library on Linux.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HF_HOME=/tmp/hf \
    # gdown/joblib write model files here at startup
    HOME=/app

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy the backend application code and the small committed artifacts.
# The two large models (churn_model_stacking.pkl, rf_model.pkl) are downloaded
# at startup by app.py via gdown.
COPY app.py .
COPY src ./src
COPY data ./data
COPY *.pkl ./

# Make the working dir writable (HF Spaces runs as a non-root user).
RUN chmod -R 777 /app

EXPOSE 7860

# Single worker + --preload so the 1GB+ of models are loaded once in the master
# and shared with the worker via copy-on-write (keeps memory in check).
# --timeout 0 disables the request timeout so the first cold model download
# (~629MB) never kills the worker.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--preload", "--timeout", "0", "app:app"]
