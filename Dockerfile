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
    # Cap OpenMP threads — cpu-basic has 2 vCPUs; also avoids xgboost/OpenMP
    # thread oversubscription on the shared runner.
    OMP_NUM_THREADS=2 \
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

# IMPORTANT: do NOT use --preload. xgboost uses OpenMP, and loading it in the
# master before gunicorn forks deadlocks the worker on first prediction
# (OpenMP + fork is unsafe). Without --preload each worker loads the models
# after forking, so OpenMP initialises cleanly in the worker process.
# One worker keeps memory in check; --timeout 300 tolerates the cold model
# download/load at boot and slow CPU inference while still recycling true hangs.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "300", "app:app"]
