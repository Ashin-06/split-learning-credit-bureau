# Runtime image: dependencies from the pinned lock, source copied in, runs as a non-root user.
FROM python:3.12-slim-bookworm AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.lock /tmp/
RUN python -m venv /opt/venv && /opt/venv/bin/pip install -r /tmp/requirements.lock

FROM python:3.12-slim-bookworm
ENV PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    SPLITBUREAU_ENV=production SPLITBUREAU_HOST=0.0.0.0 SPLITBUREAU_PORT=8017 \
    SPLITBUREAU_DATABASE_URL=sqlite:////data/splitbureau.db
RUN useradd --system --uid 10001 --no-create-home app && mkdir /data && chown app /data
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY splitbureau splitbureau
COPY web web
USER app
VOLUME /data
EXPOSE 8017
# start-period covers training the first model on an empty database (a few seconds)
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ['SPLITBUREAU_PORT'], timeout=2)"]
CMD ["python", "-m", "splitbureau", "serve"]
