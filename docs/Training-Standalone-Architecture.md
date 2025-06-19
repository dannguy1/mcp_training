# Standalone Training System Architecture

## Overview

The Standalone Training System is an independent module that consumes exported data from the MCP service and produces trained models for the inference system. It operates as a separate service with its own configuration, dependencies, and deployment process.

## Directory Structure

```
mcp_training/
├── config/
│   ├── model_config.yaml
│   └── training_config.yaml
├── docs/
│   └── Training-Standalone-Architecture.md
├── exports/
│   └── [exported_data].json
├── logs/
│   └── [log files]
├── models/
│   └── [model directories and files]
├── monitoring/
│   ├── alerts/
│   └── grafana/
│       └── dashboards/
├── scripts/
│   ├── dev.sh
│   ├── setup_dev_env.sh
│   ├── start_backend.sh
│   ├── start_frontend.sh
│   └── start_training_service.sh
├── src/
│   └── mcp_training/
│       ├── api/
│       │   ├── app.py
│       │   ├── middleware/
│       │   └── routes/
│       ├── cli.py
│       ├── core/
│       │   ├── config.py
│       │   ├── export_validator.py
│       │   ├── feature_extractor.py
│       │   ├── model_trainer.py
│       │   └── __init__.py
│       ├── models/
│       │   ├── config.py
│       │   ├── metadata.py
│       │   ├── registry.py
│       │   └── __init__.py
│       ├── services/
│       │   ├── deps.py
│       │   ├── model_service.py
│       │   ├── storage_service.py
│       │   ├── training_service.py
│       │   └── __init__.py
│       ├── utils/
│       │   ├── file_utils.py
│       │   ├── logger.py
│       │   ├── validation.py
│       │   └── __init__.py
│       └── web/
│           ├── static/
│           └── templates/
├── tests/
│   ├── conftest.py
│   ├── fixtures/
│   ├── integration/
│   └── unit/
│       ├── test_feature_extractor.py
│       └── test_model_trainer.py
├── .env
├── env.example
├── requirements.txt
├── requirements-dev.txt
├── docker-compose.yml
├── Dockerfile
└── README.md
```

## Core Components

### 1. Configuration Management
```python
# src/mcp_training/core/config.py
from pydantic import BaseSettings
import yaml

class TrainingConfig(BaseSettings):
    """Main configuration for the training system."""

class ConfigManager:
    """Configuration manager for the training system."""
```

### 2. Training Pipeline & Core Logic
- **Note:** The main training logic is implemented in `model_trainer.py`, `feature_extractor.py`, and `export_validator.py` under `src/mcp_training/core/`.
- There is **no `training_pipeline.py`**; update references accordingly.

### 3. API Server
```python
# src/mcp_training/api/app.py
from fastapi import FastAPI

def create_app() -> FastAPI:
    """Create FastAPI application."""
    return app

app = create_app()
```
- API routes and middleware are organized under `src/mcp_training/api/routes/` and `middleware/`.

### 4. CLI Interface
```python
# src/mcp_training/cli.py
import click

@click.group()
def cli():
    """MCP Training Service CLI."""

@cli.command()
def train():
    """Train a model from exported data."""

@cli.command()
def validate():
    """Validate exported data for training."""

if __name__ == '__main__':
    cli()
```

## Configuration Files

### 1. Model Configuration
```yaml
# config/model_config.yaml
# ...model configuration...
```

### 2. Training Configuration
```yaml
# config/training_config.yaml
# ...training configuration...
```

### 3. Environment Configuration
```bash
# env.example
# ...environment variables...
```

## Dependencies

### 1. Requirements
```txt
# requirements.txt
# ...Python dependencies...
```

### 2. Development Requirements
```txt
# requirements-dev.txt
# ...development dependencies...
```

---

This architecture reflects the current implementation and directory structure.  
- **Removed:** References to files not present (e.g., `training_pipeline.py`, `model_evaluator.py`, Python scripts in `scripts/`).
- **Updated:** Configuration and environment file names.
- **Added:** Noted new files (e.g., `deps.py` in services).

For further accuracy, update the configuration and CLI sections with actual code snippets if they