FROM python:3.10-slim

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# python deps
COPY requirements-api.txt /app/requirements-api.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r /app/requirements-api.txt

# app code
COPY serving /app/serving
COPY rag     /app/rag
COPY utils   /app/utils
COPY config  /app/config
# NOTE: data/ will be bind-mounted by docker-compose (see below)

# default runtime env (can be overridden)
ENV HF_HOME=/app/.cache/hf
ENV APP_CONFIG=/app/config/app.yaml

EXPOSE 8000
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
