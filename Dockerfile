# IntelSent API image (CPU-only), Python 3.10
FROM python:3.10-slim

# --- System deps (keep minimal) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# --- Python deps ---
# Copy requirements first to leverage Docker layer cache
COPY requirements-api.txt /app/requirements-api.txt

# Install deps (Torch CPU wheel works on amd64/arm64; psycopg-binary gives libpq built-in)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r /app/requirements-api.txt \
 && python - <<'PY'
# quick build-time import sanity so we fail early if something is missing
import importlib
for m in ["psycopg","numpy","torch","transformers","sentence_transformers","fastapi","uvicorn","yaml"]:
    importlib.import_module(m if m!="yaml" else "yaml")
print("Build sanity: imports OK")
PY

# --- App code ---
COPY serving /app/serving
COPY rag     /app/rag
COPY utils   /app/utils
COPY config  /app/config

# HuggingFace caches (containers can download at runtime if needed)
ENV HF_HOME=/app/.cache/hf \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers

# Expose API port
EXPOSE 8000

# Start API
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
