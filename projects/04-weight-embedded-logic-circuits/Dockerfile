FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY pyproject.toml .
RUN mkdir -p models_artifacts data
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
EXPOSE 8003
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "4"]
