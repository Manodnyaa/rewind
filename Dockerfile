FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=5

COPY backend/requirements.txt backend/requirements.txt

# CPU-only PyTorch first (~190 MB instead of ~2.5 GB of GPU libraries)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r backend/requirements.txt

# Download the embedding model at build time so startup is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY backend backend
COPY frontend frontend

WORKDIR /app/backend
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]