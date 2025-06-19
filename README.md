# MCP Training Service

A standalone machine learning training service for MCP (Model Context Protocol) data analysis, designed to process normalized logs and generate trained anomaly detection models ready for deployment on inference systems.

## 🚀 Features

- **Standalone Operation**: Runs independently on any capable host
- **WiFi Anomaly Detection**: Specialized for WiFi network log analysis
- **Multiple Model Types**: Support for Isolation Forest and Local Outlier Factor
- **RESTful API**: Complete HTTP API for training management
- **Model Management**: Versioning, deployment, and lifecycle management
- **File Upload**: Direct export file upload and processing
- **Real-time Monitoring**: Training progress tracking and logging
- **Docker Support**: Containerized deployment with Docker and Docker Compose
- **Comprehensive Testing**: Unit and integration test coverage

## 📋 Requirements

- Python 3.11+
- 4GB+ RAM (recommended)
- 2GB+ disk space for models and exports
- Docker (optional, for containerized deployment)

## 🏗️ Architecture

```
MCP Training Service
├── Core Components
│   ├── Feature Extractor - WiFi feature extraction
│   ├── Model Trainer - ML model training pipeline
│   ├── Export Validator - Data validation
│   └── Configuration Management
├── Services Layer
│   ├── Training Service - Training orchestration
│   ├── Model Service - Model management
│   └── Storage Service - File management
├── API Layer
│   ├── FastAPI Application
│   ├── RESTful Endpoints
│   └── Middleware (CORS, Auth, Logging)
└── Utilities
    ├── Logging
    ├── File Operations
    └── Validation
```

## 🚀 Quick Start

### Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mcp_training
   ```

2. **Start the service**
   ```bash
   # Production
   docker-compose up -d
   
   # Development (with hot reload)
   docker-compose --profile dev up -d
   ```

3. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health
   - Training Endpoints: http://localhost:8000/training
   - Model Endpoints: http://localhost:8000/models

### Manual Installation

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup environment**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

3. **Run the service**
   ```bash
   python -m uvicorn mcp_training.api.app:app --host 0.0.0.0 --port 8000
   ```

## 📖 Usage

### Training a Model

#### 1. Upload Export File and Train

```bash
curl -X POST "http://localhost:8000/training/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_export.json"
```

#### 2. Start Training with Existing File

```bash
curl -X POST "http://localhost:8000/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "export_file": "/path/to/export.json",
    "model_config": {
      "model_type": "isolation_forest",
      "contamination": 0.1
    }
  }'
```

#### 3. Check Training Status

```bash
curl "http://localhost:8000/training/status/{training_id}"
```

### Model Management

#### 1. List Models

```bash
curl "http://localhost:8000/models/list"
```

#### 2. Deploy a Model

```bash
curl -X POST "http://localhost:8000/models/{version}/deploy"
```

#### 3. Make Predictions

```bash
curl -X POST "http://localhost:8000/models/{version}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    "threshold": 0.5
  }'
```

## 📁 Project Structure

```
mcp_training/
├── src/mcp_training/
│   ├── api/                 # FastAPI application
│   │   ├── app.py          # Main application
│   │   ├── middleware/     # CORS, Auth, Logging
│   │   └── routes/         # API endpoints
│   ├── core/               # Core components
│   │   ├── config.py       # Configuration
│   │   ├── feature_extractor.py
│   │   ├── model_trainer.py
│   │   └── export_validator.py
│   ├── models/             # Model management
│   │   ├── config.py       # Model configuration
│   │   ├── metadata.py     # Model metadata
│   │   └── registry.py     # Model registry
│   ├── services/           # Business logic
│   │   ├── training_service.py
│   │   ├── model_service.py
│   │   └── storage_service.py
│   └── utils/              # Utilities
│       ├── logger.py       # Logging
│       ├── file_utils.py   # File operations
│       └── validation.py   # Validation
├── config/                 # Configuration files
├── tests/                  # Test suite
├── scripts/                # Utility scripts
├── Dockerfile              # Production Docker
├── docker-compose.yml      # Docker Compose
└── requirements.txt        # Dependencies
```

## ⚙️ Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Storage
MODELS_DIR=models
EXPORTS_DIR=exports
LOGS_DIR=logs

# Authentication (optional)
API_KEY=your_api_key
REQUIRE_AUTH=false
```

### Model Configuration

Edit `config/model_config.yaml`:

```yaml
model:
  type: isolation_forest
  contamination: 0.1
  random_state: 42
  n_estimators: 100

features:
  numeric:
    - signal_strength
    - bitrate
    - rx_packets
    - tx_packets
    - rx_errors
    - tx_errors

training:
  test_size: 0.2
  random_state: 42
  cross_validation_folds: 5
```

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=mcp_training

# Run specific test file
pytest tests/unit/test_model_trainer.py
```

### Test Structure

- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - Integration tests for workflows
- `tests/fixtures/` - Test data and fixtures

## 📊 Monitoring

### Health Checks

```bash
curl http://localhost:8000/health
```

### Metrics (with monitoring profile)

```bash
# Start with monitoring
docker-compose --profile monitoring up -d

# Access Prometheus
open http://localhost:9090

# Access Grafana
open http://localhost:3000
# Default credentials: admin/admin
```

## 🔧 Development

### Setup Development Environment

```bash
# Run development script
./scripts/setup_dev_env.sh

# Start development service
./scripts/start_training_service.sh
```

### Code Quality

```bash
# Format code
black src/
isort src/

# Lint code
flake8 src/
mypy src/

# Run pre-commit hooks
pre-commit run --all-files
```

## 🚀 Deployment

### Production Deployment

1. **Build and deploy with Docker**
   ```bash
   docker-compose up -d
   ```

2. **Using Docker Swarm**
   ```bash
   docker stack deploy -c docker-compose.yml mcp-training
   ```

3. **Kubernetes deployment**
   ```bash
   kubectl apply -f k8s/
   ```

### Environment-Specific Configurations

- **Development**: Use `docker-compose --profile dev`
- **Production**: Use default `docker-compose up`
- **Monitoring**: Use `docker-compose --profile monitoring`

## 📝 API Documentation

### Training Endpoints

- `POST /training/start` - Start training job
- `POST /training/upload` - Upload file and train
- `GET /training/status/{id}` - Get training status
- `GET /training/list` - List training jobs
- `DELETE /training/cancel/{id}` - Cancel training
- `GET /training/logs/{id}` - Get training logs

### Model Endpoints

- `GET /models/list` - List models
- `GET /models/{version}` - Get model info
- `POST /models/{version}/deploy` - Deploy model
- `POST /models/{version}/predict` - Make predictions
- `POST /models/{version}/evaluate` - Evaluate model
- `GET /models/{version}/download` - Download model
- `POST /models/upload` - Upload model

### Health Endpoints

- `GET /health` - Health check
- `GET /health/detailed` - Detailed health info

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check the `/docs` directory
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions

## 🔄 Version History

- **v1.0.0** - Initial release with core training functionality
- **v1.1.0** - Added model management and API improvements
- **v1.2.0** - Enhanced monitoring and deployment options

---

**Note**: This service is designed to be independent and can run on any capable host. It processes normalized logs and generates trained models ready for deployment on inference systems. 