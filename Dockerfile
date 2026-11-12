FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CITEWISE_HOST=0.0.0.0 \
    CITEWISE_INDEX_PATH=/data/index \
    FASTEMBED_CACHE_PATH=/models

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Download the embedding model at build time, so the container starts fast.
# Build with --build-arg PRELOAD_MODEL=0 to download it on first use instead.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG PRELOAD_MODEL=1
ENV CITEWISE_EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN mkdir -p /models && if [ "$PRELOAD_MODEL" = "1" ]; then \
      python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBEDDING_MODEL}')"; fi

RUN useradd --create-home app && mkdir -p /data && chown -R app /data /models
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["citewise-api"]
