#!/bin/bash

# MCP Training Service Backend Start Script
# This script manages the FastAPI backend service

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="mcp-training-backend"
PID_FILE="$PROJECT_ROOT/logs/$SERVICE_NAME.pid"
LOG_FILE="$PROJECT_ROOT/logs/$SERVICE_NAME.log"
PYTHON_PATH="$PROJECT_ROOT/venv/bin/python"
APP_MODULE="src.mcp_training.api.app:app"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to check if service is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            # PID file exists but process is dead
            rm -f "$PID_FILE"
        fi
    fi
    return 1
}

# Function to get service status
get_status() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        echo "running (PID: $pid)"
    else
        echo "stopped"
    fi
}

# Function to stop the service
stop_service() {
    print_status "Stopping $SERVICE_NAME..."
    
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            print_status "Sending SIGTERM to process $pid..."
            kill -TERM "$pid"
            
            # Wait for graceful shutdown
            local count=0
            while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 30 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            # Force kill if still running
            if ps -p "$pid" > /dev/null 2>&1; then
                print_warning "Process still running, sending SIGKILL..."
                kill -KILL "$pid"
                sleep 1
            fi
            
            rm -f "$PID_FILE"
            print_success "Service stopped"
        else
            print_warning "PID file exists but process is not running"
            rm -f "$PID_FILE"
        fi
    else
        print_warning "No PID file found, service may not be running"
    fi
}

# Function to start the service
start_service() {
    local background=${1:-false}
    
    print_status "Starting $SERVICE_NAME..."
    
    # Check if already running
    if is_running; then
        print_warning "Service is already running (PID: $(cat "$PID_FILE"))"
        return 1
    fi
    
    # Create logs directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
    mkdir -p "$(dirname "$PID_FILE")"
    
    # Check if virtual environment exists
    if [ ! -f "$PYTHON_PATH" ]; then
        print_error "Virtual environment not found at $PYTHON_PATH"
        print_error "Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        return 1
    fi
    
    # Check if required packages are installed
    if ! "$PYTHON_PATH" -c "import fastapi, uvicorn" > /dev/null 2>&1; then
        print_error "Required packages not installed"
        print_error "Please run: source venv/bin/activate && pip install -r requirements.txt"
        return 1
    fi
    
    # Set environment variables
    export PYTHONPATH="$PROJECT_ROOT"
    export MCP_TRAINING_ENV="development"
    
    if [ "$background" = true ]; then
        print_status "Starting service in background..."
        
        # Start in background
        nohup "$PYTHON_PATH" -m uvicorn "$APP_MODULE" \
            --host 0.0.0.0 \
            --port 8000 \
            --reload \
            --log-level info \
            > "$LOG_FILE" 2>&1 &
        
        local pid=$!
        echo "$pid" > "$PID_FILE"
        
        # Wait a moment to check if it started successfully
        sleep 2
        if ps -p "$pid" > /dev/null 2>&1; then
            print_success "Service started in background (PID: $pid)"
            print_status "Logs: $LOG_FILE"
            print_status "API: http://localhost:8000"
            print_status "Web UI: http://localhost:8000"
        else
            print_error "Failed to start service"
            rm -f "$PID_FILE"
            return 1
        fi
    else
        print_status "Starting service in foreground..."
        print_status "Press Ctrl+C to stop"
        
        # Start in foreground
        "$PYTHON_PATH" -m uvicorn "$APP_MODULE" \
            --host 0.0.0.0 \
            --port 8000 \
            --reload \
            --log-level info
    fi
}

# Function to restart the service
restart_service() {
    print_status "Restarting $SERVICE_NAME..."
    stop_service
    sleep 2
    start_service "$1"
}

# Function to show logs
show_logs() {
    if [ -f "$LOG_FILE" ]; then
        if command -v tail > /dev/null 2>&1; then
            print_status "Showing last 50 lines of logs (Press Ctrl+C to exit):"
            tail -f -n 50 "$LOG_FILE"
        else
            print_status "Log file: $LOG_FILE"
            cat "$LOG_FILE"
        fi
    else
        print_warning "No log file found"
    fi
}

# Function to show help
show_help() {
    echo "MCP Training Service Backend Management Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start     Start the backend service"
    echo "  stop      Stop the backend service"
    echo "  restart   Restart the backend service"
    echo "  status    Show service status"
    echo "  logs      Show service logs"
    echo "  help      Show this help message"
    echo ""
    echo "Options:"
    echo "  -b, --background  Start in background mode"
    echo "  -f, --foreground  Start in foreground mode (default)"
    echo ""
    echo "Examples:"
    echo "  $0 start              # Start in foreground"
    echo "  $0 start -b           # Start in background"
    echo "  $0 stop               # Stop the service"
    echo "  $0 restart -b         # Restart in background"
    echo "  $0 status             # Show status"
    echo "  $0 logs               # Show logs"
}

# Main script logic
case "${1:-help}" in
    start)
        background=false
        if [ "$2" = "-b" ] || [ "$2" = "--background" ]; then
            background=true
        fi
        start_service "$background"
        ;;
    stop)
        stop_service
        ;;
    restart)
        background=false
        if [ "$2" = "-b" ] || [ "$2" = "--background" ]; then
            background=true
        fi
        restart_service "$background"
        ;;
    status)
        print_status "Service status: $(get_status)"
        if is_running; then
            local pid=$(cat "$PID_FILE")
            print_status "Process info:"
            ps -p "$pid" -o pid,ppid,cmd,etime
        fi
        ;;
    logs)
        show_logs
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac 