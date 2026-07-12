# syntax=docker/dockerfile:1

FROM python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf AS runtime

ARG RELEASE_COMMIT_SHA=""

LABEL org.opencontainers.image.title="MBZUAI retrieval and indexing pipeline" \
      org.opencontainers.image.revision="${RELEASE_COMMIT_SHA}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    RELEASE_COMMIT_SHA="${RELEASE_COMMIT_SHA}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libgomp1 \
    libmagic1 \
    poppler-utils \
    tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY pipeline/requirements-docling.txt /app/pipeline/requirements-docling.txt
RUN python -m pip install pip==26.1.2 setuptools==82.0.1 wheel==0.47.0 \
    && python -m pip install -r /app/requirements.txt \
    && python -m playwright install --with-deps chromium \
    && rm -rf /root/.cache/pip

COPY . /app

RUN mkdir -p /data/releases /data/cache \
    && chmod +x /app/scripts/deploy/*.sh

VOLUME ["/data/releases", "/data/cache"]

EXPOSE 8060

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=8 \
  CMD curl -fsS http://127.0.0.1:8060/readyz >/dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/scripts/deploy/start-retriever-from-active-release.sh"]
