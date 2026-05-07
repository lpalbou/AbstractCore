ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ARG ABSTRACTCORE_VERSION

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000

RUN addgroup --system abstractcore \
    && adduser --system --ingroup abstractcore --home /home/abstractcore abstractcore \
    && python -m pip install --upgrade pip \
    && ABSTRACTCORE_WHEEL_URL="$(python -c 'import json, os, urllib.request; version = os.environ["ABSTRACTCORE_VERSION"]; url = "https://pypi.org/pypi/abstractcore/" + version + "/json"; payload = json.load(urllib.request.urlopen(url, timeout=20)); print(next(item["url"] for item in payload["urls"] if item.get("packagetype") == "bdist_wheel" and item.get("filename", "").endswith(".whl")))')" \
    && python -m pip install "abstractcore[server,remote,media,tokens,compression] @ ${ABSTRACTCORE_WHEEL_URL}"

USER abstractcore
WORKDIR /home/abstractcore

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\", \"8000\")}/health', timeout=3).read()" || exit 1

CMD ["sh", "-c", "python -m uvicorn abstractcore.server.app:app --host \"${HOST:-0.0.0.0}\" --port \"${PORT:-8000}\""]
