# Lightweight base
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# (Optional) system deps for building some wheels faster
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only API deps (no heavy ML stuff needed for the API container)
COPY requirements-api.txt /app/
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy just the code needed by the API
COPY serving /app/serving
COPY rag /app/rag
COPY utils /app/utils
COPY config /app/config

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
