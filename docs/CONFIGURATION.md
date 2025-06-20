# Configuration Management

The MCP Training Service now uses a unified configuration system based on environment variables and `.env` files. This provides better flexibility, security, and deployment options.

## Overview

All configuration settings are now managed through environment variables, with the following benefits:

- **Environment-specific configuration**: Different settings for development, staging, and production
- **Security**: Sensitive settings can be managed through environment variables
- **Deployment flexibility**: Easy to configure in containers and cloud environments
- **Version control**: Configuration templates can be versioned without exposing sensitive data

## Configuration Sources

The application loads configuration in the following order (later sources override earlier ones):

1. **Default values** (hardcoded in `TrainingConfig`)
2. **Environment variables** (system environment)
3. **`.env` file** (local development)
4. **Runtime updates** (via Settings API)

## Configuration Categories

### Service Configuration
- `TRAINING_SERVICE_NAME`: Display name for the service
- `TRAINING_SERVICE_VERSION`: Service version
- `TRAINING_DEBUG`: Enable debug mode

### API Configuration
- `TRAINING_API_HOST`: API server host (default: 0.0.0.0)
- `TRAINING_API_PORT`: API server port (default: 8000)
- `TRAINING_API_WORKERS`: Number of worker processes

### Storage Configuration
- `TRAINING_MODELS_DIR`: Directory for trained models
- `TRAINING_EXPORTS_DIR`: Directory for export files
- `TRAINING_LOGS_DIR`: Directory for log files
- `TRAINING_MAX_STORAGE_GB`: Maximum storage usage in GB
- `TRAINING_AUTO_CLEANUP`: Enable automatic cleanup
- `TRAINING_RETENTION_DAYS`: Days to keep old files

### Training Configuration
- `TRAINING_MAX_TRAINING_TIME`: Maximum training time in seconds
- `TRAINING_MAX_MEMORY_USAGE`: Maximum memory usage in MB
- `TRAINING_ENABLE_GPU`: Enable GPU acceleration
- `TRAINING_MAX_CONCURRENT_JOBS`: Maximum concurrent training jobs
- `TRAINING_DEFAULT_MAX_ITERATIONS`: Default training iterations
- `TRAINING_DEFAULT_LEARNING_RATE`: Default learning rate
- `TRAINING_JOB_TIMEOUT`: Job timeout in hours

### Logging Configuration
- `TRAINING_LOG_LEVEL`: Log level (DEBUG, INFO, WARNING, ERROR)
- `TRAINING_LOG_FILE`: Log file path
- `TRAINING_LOG_FORMAT`: Log format (structured, simple, detailed)
- `TRAINING_MAX_LOG_SIZE_MB`: Maximum log file size
- `TRAINING_LOG_TO_CONSOLE`: Log to console
- `TRAINING_LOG_TO_FILE`: Log to file

### General UI Configuration
- `TRAINING_TIMEZONE`: Application timezone
- `TRAINING_DATE_FORMAT`: Date format for display
- `TRAINING_AUTO_REFRESH`: Enable auto-refresh on dashboard
- `TRAINING_NOTIFICATIONS`: Enable browser notifications

### Security Configuration
- `TRAINING_AUTH_ENABLED`: Enable authentication
- `TRAINING_API_KEY`: API key for authentication
- `TRAINING_CORS_ORIGINS`: Allowed CORS origins (comma-separated)
- `TRAINING_RATE_LIMIT`: Rate limit (requests per minute)
- `TRAINING_HTTPS_ONLY`: Require HTTPS
- `TRAINING_SECURE_HEADERS`: Enable security headers

### Advanced Configuration
- `TRAINING_ENABLE_MONITORING`: Enable monitoring
- `TRAINING_PROMETHEUS_PORT`: Prometheus metrics port
- `TRAINING_PERFORMANCE_MONITORING`: Enable performance monitoring
- `TRAINING_WEBSOCKET_ENABLED`: Enable WebSocket connections
- `TRAINING_AUTO_BACKUP`: Enable automatic backups

## Setup Instructions

### 1. Initial Setup

For new installations, copy the example configuration:

```bash
cp env.example .env
```

### 2. Migration from Old Settings

If you have existing settings in `config/user_settings.json`, run the migration script:

```bash
python3 scripts/migrate_settings.py
```

This will:
- Convert existing settings to environment variables
- Create a `.env` file
- Backup original settings to `config/user_settings.json.backup`

### 3. Customization

Edit the `.env` file to customize settings for your environment:

```bash
# Edit the .env file
nano .env

# Or use your preferred editor
code .env
```

### 4. Environment-Specific Configuration

For different environments, you can use different `.env` files:

```bash
# Development
cp .env .env.development

# Production
cp .env .env.production

# Load specific environment
export ENV_FILE=.env.production
```

## Runtime Configuration Updates

The Settings page in the web interface allows you to update configuration at runtime. Changes are automatically saved to the `.env` file and take effect immediately.

### API Endpoints

- `GET /api/settings/`: Get current configuration
- `PUT /api/settings/`: Update configuration

## Environment Variables in Production

For production deployments, set environment variables directly:

```bash
# Docker
docker run -e TRAINING_API_PORT=8000 -e TRAINING_LOG_LEVEL=INFO ...

# Kubernetes
env:
  - name: TRAINING_API_PORT
    value: "8000"
  - name: TRAINING_LOG_LEVEL
    value: "INFO"

# System environment
export TRAINING_API_PORT=8000
export TRAINING_LOG_LEVEL=INFO
```

## Configuration Validation

The configuration system includes validation for:

- Data types (int, float, bool, str)
- Value ranges (e.g., port numbers, timeouts)
- Required fields
- Format validation (e.g., CORS origins)

## Troubleshooting

### Common Issues

1. **Settings not updating**: Restart the application after changing `.env` file
2. **Permission errors**: Ensure the application can read/write the `.env` file
3. **Invalid values**: Check the configuration validation in the logs

### Debug Configuration

Enable debug mode to see detailed configuration loading:

```bash
TRAINING_DEBUG=true
```

### Configuration Dump

To see the current configuration, use the API:

```bash
curl http://localhost:8000/api/settings/
```

## Best Practices

1. **Never commit `.env` files** with sensitive data to version control
2. **Use different configurations** for different environments
3. **Validate configuration** before deployment
4. **Document custom settings** for your deployment
5. **Use secrets management** for sensitive data in production

## Migration Notes

The new configuration system is backward compatible. Existing functionality will continue to work, but you should migrate to the new system for better management and flexibility.

For questions or issues with configuration, please refer to the troubleshooting section or create an issue in the project repository. 