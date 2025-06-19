# MCP Training Service - Quick Start Guide

## Prerequisites

- Python 3.11 or higher
- Git
- Docker (optional)

## Installation

1. **Clone or navigate to the training directory**
   ```bash
   cd /home/dannguyen/WNC/mcp_training
   ```

2. **Run the setup script**
   ```bash
   ./scripts/start_training_service.sh
   ```

   This will:
   - Check Python version
   - Create virtual environment
   - Install dependencies
   - Create necessary directories
   - Set up environment file

3. **Review and edit configuration (optional)**
   ```bash
   # Edit environment settings
   nano .env
   
   # Edit model configuration
   nano config/model_config.yaml
   
   # Edit training service configuration
   nano config/training_config.yaml
   ```

## Quick Test

1. **Validate the sample export**
   ```bash
   python -m mcp_training.cli validate exports/sample_export.json
   ```

2. **Train a model from the sample data**
   ```bash
   python -m mcp_training.cli train exports/sample_export.json
   ```

3. **List trained models**
   ```bash
   python -m mcp_training.cli list-models
   ```

4. **Make predictions**
   ```bash
   python -m mcp_training.cli predict exports/sample_export.json <model_name>
   ```

## API Server

Start the API server:
```bash
./scripts/start_training_service.sh --api
```

Or manually:
```bash
uvicorn mcp_training.api.app:app --host 0.0.0.0 --port 8001 --reload
```

Access the API documentation at: http://localhost:8001/docs

## Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

2. **Train a model via Docker**
   ```bash
   docker exec mcp-training python -m mcp_training.cli train exports/sample_export.json
   ```

## Integration with MCP Service

1. **Export data from MCP service**
   ```bash
   # In the main MCP service directory
   curl -X POST "http://localhost:8000/api/v1/export/export" \
     -H "Content-Type: application/json" \
     -d '{"format": "json", "include_metadata": true}'
   ```

2. **Copy export to training service**
   ```bash
   cp /path/to/mcp_service/exports/export_*.json /home/dannguyen/WNC/mcp_training/exports/
   ```

3. **Train model on the export**
   ```bash
   python -m mcp_training.cli train exports/export_<timestamp>.json
   ```

4. **Copy trained model back to MCP service**
   ```bash
   cp -r models/<model_name> /path/to/mcp_service/models/
   ```

## Testing

Run the test suite:
```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=mcp_training

# Run specific test
pytest tests/unit/test_feature_extractor.py
```

## Monitoring (Optional)

Start monitoring services:
```bash
docker-compose --profile monitoring up -d
```

Access:
- Prometheus: http://localhost:9091
- Grafana: http://localhost:3000 (admin/admin)

## Troubleshooting

### Common Issues

1. **Python version error**
   ```bash
   # Check Python version
   python3 --version
   
   # Install Python 3.11 if needed
   sudo apt update
   sudo apt install python3.11 python3.11-venv
   ```

2. **Permission denied on script**
   ```bash
   chmod +x scripts/start_training_service.sh
   ```

3. **Import errors**
   ```bash
   # Ensure PYTHONPATH is set
   export PYTHONPATH=$PWD/src
   
   # Or activate virtual environment
   source venv/bin/activate
   ```

4. **Port already in use**
   ```bash
   # Check what's using port 8001
   lsof -i :8001
   
   # Kill process or change port in .env
   ```

### Getting Help

- Check logs: `tail -f logs/training.log`
- API health: `curl http://localhost:8001/health`
- System info: `python -m mcp_training.cli info`

## Next Steps

1. **Customize feature extraction** in `config/model_config.yaml`
2. **Add more model types** in the ModelTrainer class
3. **Implement custom evaluation metrics**
4. **Set up automated training pipelines**
5. **Integrate with CI/CD systems**

## File Structure

```
mcp_training/
├── config/                 # Configuration files
├── src/mcp_training/       # Source code
│   ├── api/               # FastAPI server
│   ├── core/              # Core components
│   └── cli.py             # CLI interface
├── scripts/               # Utility scripts
├── tests/                 # Test suite
├── models/                # Trained models
├── exports/               # Export data
├── logs/                  # Log files
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose
└── requirements.txt       # Dependencies
``` 