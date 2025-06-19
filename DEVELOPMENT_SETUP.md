# MCP Training Service Development Setup

This guide will help you set up and run the MCP Training Service for development and debugging.

## 🚀 Quick Start

### 1. Setup Development Environment
```bash
# Setup virtual environment and install dependencies
./scripts/dev.sh setup
```

### 2. Start All Services
```bash
# Start in foreground (recommended for development)
./scripts/dev.sh start

# Or start in background
./scripts/dev.sh start -b
```

### 3. Access the Application
- **Web UI**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Backend API**: http://localhost:8000/api
- **Frontend Assets**: http://localhost:3000

### 4. Stop Services
```bash
# Stop all services
./scripts/dev.sh stop
```

## 📋 Available Scripts

### Main Development Script (`./scripts/dev.sh`)
The primary script for managing the entire development environment.

**Key Commands:**
- `setup` - Initialize development environment
- `start` - Start all services
- `stop` - Stop all services
- `restart` - Restart all services
- `status` - Check service status
- `logs` - View service logs
- `cleanup` - Clean up environment

### Individual Service Scripts

#### Backend Script (`./scripts/start_backend.sh`)
Manages the FastAPI backend service.

```bash
# Start backend
./scripts/start_backend.sh start

# Start in background
./scripts/start_backend.sh start -b

# Check status
./scripts/start_backend.sh status

# View logs
./scripts/start_backend.sh logs
```

#### Frontend Script (`./scripts/start_frontend.sh`)
Manages the frontend development server.

```bash
# Start frontend
./scripts/start_frontend.sh start

# Start on custom port
./scripts/start_frontend.sh start -p 3001

# Watch for file changes
./scripts/start_frontend.sh watch
```

## 🔧 Development Workflow

### Daily Development
```bash
# 1. Start all services
./scripts/dev.sh start

# 2. Make your code changes...

# 3. View logs if needed
./scripts/dev.sh logs backend
./scripts/dev.sh logs frontend

# 4. Stop services when done
./scripts/dev.sh stop
```

### Debugging
```bash
# Check if services are running
./scripts/dev.sh status

# View specific service logs
./scripts/dev.sh logs backend
./scripts/dev.sh logs frontend

# Restart services if needed
./scripts/dev.sh restart
```

## 🏗️ Project Structure

```
mcp_training/
├── scripts/                    # Development scripts
│   ├── dev.sh                 # Main development script
│   ├── start_backend.sh       # Backend management
│   ├── start_frontend.sh      # Frontend management
│   └── README.md              # Script documentation
├── src/mcp_training/
│   ├── api/                   # FastAPI application
│   │   ├── app.py            # Main FastAPI app
│   │   └── routes/           # API routes
│   ├── web/                  # Web interface
│   │   ├── static/           # CSS, JS, assets
│   │   └── templates/        # HTML templates
│   ├── core/                 # Core business logic
│   ├── services/             # Service layer
│   └── models/               # Data models
├── logs/                     # Service logs
├── models/                   # Trained models
├── exports/                  # Export files
└── requirements.txt          # Python dependencies
```

## 🌐 Service Architecture

The MCP Training Service consists of:

1. **Backend (FastAPI)**: 
   - Serves both API endpoints and web pages
   - Handles training jobs, model management
   - Provides real-time updates via WebSocket
   - Runs on port 8000

2. **Frontend (Static Assets)**:
   - Bootstrap-based responsive UI
   - Real-time dashboard with charts
   - Training management interface
   - Model management interface
   - Served by backend on port 8000

## 🔍 Debugging Features

### Log Management
```bash
# View all logs
./scripts/dev.sh logs all

# View specific service logs
./scripts/dev.sh logs backend
./scripts/dev.sh logs frontend

# Follow logs in real-time
./scripts/dev.sh logs backend | tail -f
```

### Health Checks
```bash
# Check backend health
curl http://localhost:8000/api/health

# Check service status
./scripts/dev.sh status
```

### Process Management
```bash
# View running processes
ps aux | grep mcp

# Check port usage
netstat -tulpn | grep :8000
lsof -i :8000
```

## 🛠️ Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# 1. Check virtual environment
ls venv/

# 2. Check dependencies
pip list | grep fastapi

# 3. Check logs
./scripts/dev.sh logs all

# 4. Check port availability
lsof -i :8000
```

#### Port Already in Use
```bash
# Find and kill process using port 8000
lsof -i :8000
kill -9 <PID>

# Or restart services
./scripts/dev.sh restart
```

#### Virtual Environment Issues
```bash
# Recreate virtual environment
rm -rf venv
./scripts/dev.sh setup
```

#### Permission Issues
```bash
# Make scripts executable
chmod +x scripts/*.sh
```

### Performance Issues
```bash
# Check system resources
htop
df -h
free -h

# Check service performance
./scripts/dev.sh logs backend | grep "slow"
```

## 📊 Monitoring

### Service Status
```bash
# Real-time status
./scripts/dev.sh status

# Process information
ps aux | grep mcp
```

### Log Monitoring
```bash
# Monitor backend logs
tail -f logs/mcp-training-backend.log

# Monitor frontend logs
tail -f logs/mcp-training-frontend.log
```

### API Health
```bash
# Health check
curl http://localhost:8000/api/health

# API documentation
open http://localhost:8000/docs
```

## 🔄 Development Tips

### Hot Reloading
- Backend automatically reloads on code changes
- Frontend assets are served with cache busting
- Use `./scripts/start_frontend.sh watch` for file watching

### Code Changes
1. Make changes to your code
2. Backend will auto-reload (if running in foreground)
3. Refresh browser to see frontend changes
4. Check logs for any errors

### Testing
```bash
# Run tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_feature_extractor.py -v
```

### Database/Storage
- Models are stored in `models/` directory
- Exports are stored in `exports/` directory
- Logs are stored in `logs/` directory

## 🚀 Production Deployment

For production deployment, see the main `README.md` file for Docker and deployment instructions.

## 📚 Additional Resources

- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [UI Design Guide](docs/UI-Design-Guide.md) - Frontend design specifications
- [Implementation Plan](docs/Implementation-Plan.md) - Project roadmap
- [Scripts Documentation](scripts/README.md) - Detailed script usage

## 🤝 Contributing

1. Follow the development workflow above
2. Make changes in feature branches
3. Test thoroughly before submitting
4. Update documentation as needed
5. Follow the coding standards in the project

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the logs: `./scripts/dev.sh logs all`
3. Check the service status: `./scripts/dev.sh status`
4. Consult the documentation in the `docs/` directory
5. Create an issue with detailed error information 