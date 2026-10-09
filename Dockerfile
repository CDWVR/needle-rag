FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NEEDLE_DATA_DIR=/data \
    HOME=/home/needle

RUN useradd --create-home --uid 10001 needle && mkdir -p /data /app && chown needle:needle /data /app

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

# Download the local embedding model into the image so the first upload does not have to.
USER needle
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warm up'])"
USER root

COPY backend backend
COPY frontend frontend
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

WORKDIR /app/backend
# No VOLUME instruction: Railway rejects it. Attach a Railway volume at /data instead (see docs/deploy-railway.md).
EXPOSE 8000
ENTRYPOINT ["docker-entrypoint.sh"]
# --proxy-headers: rate limits must key on the real client, not the platform's proxy.
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
