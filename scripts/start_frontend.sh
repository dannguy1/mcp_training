#!/bin/bash

# MCP Training Service Frontend Development Script
# This script manages the frontend development server with hot reloading

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="mcp-training-frontend"
PID_FILE="$PROJECT_ROOT/logs/$SERVICE_NAME.pid"
LOG_FILE="$PROJECT_ROOT/logs/$SERVICE_NAME.log"
FRONTEND_DIR="$PROJECT_ROOT/src/mcp_training/web"
BACKEND_URL="http://localhost:8000"

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

# Function to check if backend is running
check_backend() {
    if command -v curl > /dev/null 2>&1; then
        if curl -s "$BACKEND_URL/api/health" > /dev/null 2>&1; then
            return 0
        fi
    elif command -v wget > /dev/null 2>&1; then
        if wget -q --spider "$BACKEND_URL/api/health" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# Function to start the service
start_service() {
    local background=${1:-false}
    local port=${2:-3000}
    
    print_status "Starting $SERVICE_NAME..."
    
    # Check if already running
    if is_running; then
        print_warning "Service is already running (PID: $(cat "$PID_FILE"))"
        return 1
    fi
    
    # Check if backend is running (warn only)
    if ! check_backend; then
        print_warning "Backend service is not running at $BACKEND_URL"
        print_warning "You may need to start the backend for full functionality: ./scripts/start_backend.sh start"
        # Do not exit; continue to start static server
    fi
    
    # Create logs directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"
    mkdir -p "$(dirname "$PID_FILE")"
    
    # Check if frontend directory exists
    if [ ! -d "$FRONTEND_DIR" ]; then
        print_error "Frontend directory not found: $FRONTEND_DIR"
        return 1
    fi
    
    # Check if Python is available for serving static files
    if ! command -v python3 > /dev/null 2>&1; then
        print_error "Python3 is required but not found"
        return 1
    fi
    
    # Set environment variables
    export FRONTEND_PORT="$port"
    export BACKEND_URL="$BACKEND_URL"
    
    if [ "$background" = true ]; then
        print_status "Starting frontend development server in background..."
        
        # Start Python HTTP server in background
        cd "$FRONTEND_DIR/static"
        nohup python3 -m http.server "$port" \
            --bind 0.0.0.0 \
            > "$LOG_FILE" 2>&1 &
        
        local pid=$!
        echo "$pid" > "$PID_FILE"
        
        # Wait a moment to check if it started successfully
        sleep 2
        if ps -p "$pid" > /dev/null 2>&1; then
            print_success "Frontend development server started in background (PID: $pid)"
            print_status "Logs: $LOG_FILE"
            print_status "Frontend: http://localhost:$port"
            print_status "Backend API: $BACKEND_URL"
            print_status "Web UI: $BACKEND_URL"
        else
            print_error "Failed to start frontend development server"
            rm -f "$PID_FILE"
            return 1
        fi
    else
        print_status "Starting frontend development server in foreground..."
        print_status "Press Ctrl+C to stop"
        print_status "Frontend: http://localhost:$port"
        print_status "Backend API: $BACKEND_URL"
        print_status "Web UI: $BACKEND_URL"
        
        # Start in foreground
        cd "$FRONTEND_DIR/static"
        python3 -m http.server "$port" --bind 0.0.0.0
    fi
}

# Function to restart the service
restart_service() {
    print_status "Restarting $SERVICE_NAME..."
    stop_service
    sleep 2
    start_service "$1" "$2"
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

# Function to watch and rebuild frontend assets
watch_assets() {
    print_status "Starting asset watcher..."
    
    # Check if inotify-tools is available (Linux)
    if command -v inotifywait > /dev/null 2>&1; then
        print_status "Using inotify for file watching..."
        watch_with_inotify
    elif command -v fswatch > /dev/null 2>&1; then
        print_status "Using fswatch for file watching..."
        watch_with_fswatch
    else
        print_warning "No file watcher available. Install inotify-tools (Linux) or fswatch (macOS)"
        print_status "Manual refresh required when files change"
    fi
}

# Function to watch files with inotify (Linux)
watch_with_inotify() {
    local css_dir="$FRONTEND_DIR/static/css"
    local js_dir="$FRONTEND_DIR/static/js"
    
    inotifywait -m -r -e modify,create,delete "$css_dir" "$js_dir" | while read path action file; do
        print_status "File change detected: $path$file ($action)"
        print_status "Frontend assets updated"
    done
}

# Function to watch files with fswatch (macOS)
watch_with_fswatch() {
    local css_dir="$FRONTEND_DIR/static/css"
    local js_dir="$FRONTEND_DIR/static/js"
    
    fswatch -o "$css_dir" "$js_dir" | while read f; do
        print_status "Frontend assets updated"
    done
}

# Function to show help
show_help() {
    echo "MCP Training Service Frontend Development Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start     Start the frontend development server"
    echo "  stop      Stop the frontend development server"
    echo "  restart   Restart the frontend development server"
    echo "  status    Show service status"
    echo "  logs      Show service logs"
    echo "  watch     Watch for file changes and rebuild assets"
    echo "  help      Show this help message"
    echo ""
    echo "Options:"
    echo "  -b, --background  Start in background mode"
    echo "  -f, --foreground  Start in foreground mode (default)"
    echo "  -p, --port PORT   Specify port (default: 3000)"
    echo ""
    echo "Examples:"
    echo "  $0 start              # Start in foreground"
    echo "  $0 start -b           # Start in background"
    echo "  $0 start -p 3001      # Start on port 3001"
    echo "  $0 stop               # Stop the service"
    echo "  $0 restart -b         # Restart in background"
    echo "  $0 status             # Show status"
    echo "  $0 logs               # Show logs"
    echo "  $0 watch              # Watch for file changes"
    echo ""
    echo "Note: This script starts a simple HTTP server for frontend assets."
    echo "The main web interface is served by the backend at http://localhost:8000"
}

# Main script logic
case "${1:-help}" in
    start)
        background=false
        port=3000
        
        # Parse options
        shift
        while [[ $# -gt 0 ]]; do
            case $1 in
                -b|--background)
                    background=true
                    shift
                    ;;
                -f|--foreground)
                    background=false
                    shift
                    ;;
                -p|--port)
                    port="$2"
                    shift 2
                    ;;
                *)
                    print_error "Unknown option: $1"
                    exit 1
                    ;;
            esac
        done
        
        start_service "$background" "$port"
        ;;
    stop)
        stop_service
        ;;
    restart)
        background=false
        port=3000
        
        # Parse options
        shift
        while [[ $# -gt 0 ]]; do
            case $1 in
                -b|--background)
                    background=true
                    shift
                    ;;
                -f|--foreground)
                    background=false
                    shift
                    ;;
                -p|--port)
                    port="$2"
                    shift 2
                    ;;
                *)
                    print_error "Unknown option: $1"
                    exit 1
                    ;;
            esac
        done
        
        restart_service "$background" "$port"
        ;;
    status)
        print_status "Service status: $(get_status)"
        if is_running; then
            local pid=$(cat "$PID_FILE")
            print_status "Process info:"
            ps -p "$pid" -o pid,ppid,cmd,etime
        fi
        print_status "Backend status: $(check_backend && echo "running" || echo "stopped")"
        ;;
    logs)
        show_logs
        ;;
    watch)
        watch_assets
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