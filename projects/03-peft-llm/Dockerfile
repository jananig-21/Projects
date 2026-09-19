FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY pyproject.toml .
RUN mkdir -p models_artifacts
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
EXPOSE 8002
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
