# MCP Training Service Development Scripts

This directory contains scripts to manage the MCP Training Service development environment.

## Quick Start

### 1. Setup Development Environment
```bash
./scripts/dev.sh setup
```

### 2. Start All Services
```bash
# Start in foreground (recommended for development)
./scripts/dev.sh start

# Start in background
./scripts/dev.sh start -b
```

### 3. Stop All Services
```bash
./scripts/dev.sh stop
```

## Available Scripts

### Main Development Script (`dev.sh`)
The main script that manages both backend and frontend services.

**Commands:**
- `start` - Start all development services
- `stop` - Stop all development services  
- `restart` - Restart all development services
- `status` - Show status of all services
- `logs` - Show service logs
- `setup` - Setup development environment
- `cleanup` - Clean up development environment
- `help` - Show help message

**Options:**
- `-b, --background` - Start services in background mode
- `-f, --foreground` - Start services in foreground mode (default)

**Examples:**
```bash
# Setup environment
./scripts/dev.sh setup

# Start in foreground (Ctrl+C to stop)
./scripts/dev.sh start

# Start in background
./scripts/dev.sh start -b

# Check status
./scripts/dev.sh status

# View logs
./scripts/dev.sh logs backend
./scripts/dev.sh logs frontend
./scripts/dev.sh logs all

# Stop all services
./scripts/dev.sh stop

# Restart all services
./scripts/dev.sh restart
```

### Backend Script (`start_backend.sh`)
Manages the FastAPI backend service.

**Commands:**
- `start` - Start the backend service
- `stop` - Stop the backend service
- `restart` - Restart the backend service
- `status` - Show service status
- `logs` - Show service logs
- `help` - Show help message

**Examples:**
```bash
# Start backend in foreground
./scripts/start_backend.sh start

# Start backend in background
./scripts/start_backend.sh start -b

# Check backend status
./scripts/start_backend.sh status

# View backend logs
./scripts/start_backend.sh logs
```

### Frontend Script (`start_frontend.sh`)
Manages the frontend development server.

**Commands:**
- `start` - Start the frontend service
- `stop` - Stop the frontend service
- `restart` - Restart the frontend service
- `status` - Show service status
- `logs` - Show service logs
- `watch` - Watch for file changes
- `help` - Show help message

**Options:**
- `-p, --port PORT` - Specify port (default: 3000)

**Examples:**
```bash
# Start frontend on default port (3000)
./scripts/start_frontend.sh start

# Start frontend on custom port
./scripts/start_frontend.sh start -p 3001

# Watch for file changes
./scripts/start_frontend.sh watch
```

## Service URLs

When all services are running:

- **Backend API**: http://localhost:8000
- **Web UI**: http://localhost:8000
- **Frontend Assets**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs

## Development Workflow

### 1. Initial Setup
```bash
# Clone the repository
git clone <repository-url>
cd mcp_training

# Setup development environment
./scripts/dev.sh setup
```

### 2. Daily Development
```bash
# Start all services
./scripts/dev.sh start

# Make changes to code...

# View logs if needed
./scripts/dev.sh logs backend

# Stop services when done
./scripts/dev.sh stop
```

### 3. Debugging
```bash
# Check service status
./scripts/dev.sh status

# View specific service logs
./scripts/dev.sh logs backend
./scripts/dev.sh logs frontend

# Restart services if needed
./scripts/dev.sh restart
```

## Troubleshooting

### Service Won't Start
1. Check if virtual environment exists: `ls venv/`
2. Check if dependencies are installed: `pip list`
3. Check logs: `./scripts/dev.sh logs all`
4. Check port availability: `netstat -tulpn | grep :8000`

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or restart services
./scripts/dev.sh restart
```

### Virtual Environment Issues
```bash
# Remove and recreate virtual environment
rm -rf venv
./scripts/dev.sh setup
```

### Permission Issues
```bash
# Make scripts executable
chmod +x scripts/*.sh
```

## File Structure

```
scripts/
├── dev.sh                 # Main development script
├── start_backend.sh       # Backend service management
├── start_frontend.sh      # Frontend service management
├── setup_dev_env.sh       # Environment setup
├── start_training_service.sh  # Training service management
└── README.md              # This file
```

## Environment Variables

The scripts automatically set these environment variables:

- `PYTHONPATH` - Set to project root
- `MCP_TRAINING_ENV` - Set to "development"
- `FRONTEND_PORT` - Frontend server port
- `BACKEND_URL` - Backend API URL

## Log Files

Log files are stored in the `logs/` directory:

- `logs/mcp-training-backend.log` - Backend service logs
- `logs/mcp-training-frontend.log` - Frontend service logs
- `logs/mcp-training-backend.pid` - Backend process ID
- `logs/mcp-training-frontend.pid` - Frontend process ID

## Requirements

- Python 3.8+
- Bash shell
- curl or wget (for health checks)
- inotify-tools (Linux) or fswatch (macOS) for file watching (optional) 