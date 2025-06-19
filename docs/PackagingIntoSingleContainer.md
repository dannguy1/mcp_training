# Packaging the Standalone Training System into a Single Container (Web UI Only)

This document describes how to package the Standalone Training System into a single Docker container that exposes only the web UI (API server). The CLI, Prometheus, and Grafana are not included. This approach enables simple, portable deployment for development or production.

---

## 1. Project Structure Review

The project contains:
- `Dockerfile` for containerization
- `src/` for all source code (including FastAPI app and web UI)
- `config/` for YAML configuration files
- `models/`, `exports/`, and `logs/` for data, model artifacts, and logs
- `.env` for environment variables

---

## 2. Requirements

- Only the FastAPI web server (and optionally static files/templates for UI) will run in the container.
- No CLI entrypoint.
- No monitoring/metrics stack.

---

## 3. Dockerfile Example

Update or create your `Dockerfile` as follows:

````dockerfile
# filepath: Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY [requirements.txt](http://_vscodecontentref_/0) .
RUN pip install --no-cache-dir -r [requirements.txt](http://_vscodecontentref_/1)

# Copy application code and configs
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY exports/ ./exports/
COPY logs/ ./logs/

# Copy web static and templates if needed
COPY src/mcp_training/web/static/ ./src/mcp_training/web/static/
COPY src/mcp_training/web/templates/ ./src/mcp_training/web/templates/

# Copy .env if needed (or mount at runtime)
COPY .env .env

EXPOSE 8000

# Entrypoint: Start FastAPI app
CMD ["uvicorn", "src.mcp_training.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
````