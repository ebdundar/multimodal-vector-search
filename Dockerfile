# Multi-stage Dockerfile for FastAPI app (main:app)

# ---- Build stage ----
FROM python:3.11-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --prefix=/install -r requirements.txt

# ---- Final stage ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Copy installed packages from build stage
COPY --from=build /install /usr/local

# Copy application files
COPY . /app

# Ensure logs go to stdout/stderr
ENV PATH="/usr/local/bin:$PATH"

EXPOSE 8080


CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "300", "--graceful-timeout", "60", "--access-logfile", "-", "--error-logfile", "-", "-b", "0.0.0.0:8080", "main:app"]

