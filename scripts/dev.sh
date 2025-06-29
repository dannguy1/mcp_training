#!/bin/bash

# MCP Training Service Development Script
# This script manages both backend and frontend development servers

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_SCRIPT="$SCRIPT_DIR/start_backend.sh"
FRONTEND_SCRIPT="$SCRIPT_DIR/start_frontend.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

print_info() {
    echo -e "${CYAN}ℹ${NC} $1"
}

# Function to check if a service is running
check_service_status() {
    local service_name="$1"
    local pid_file="$PROJECT_ROOT/logs/$service_name.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "running (PID: $pid)"
            return 0
        else
            rm -f "$pid_file"
        fi
    fi
    echo "stopped"
    return 1
}

# Function to start all services
start_all() {
    local background=${1:-false}
    
    print_status "Starting MCP Training Service development environment..."
    
    # Start backend first
    print_info "Starting backend service..."
    if [ "$background" = true ]; then
        "$BACKEND_SCRIPT" start -b
    else
        "$BACKEND_SCRIPT" start
    fi
    
    # Wait for backend to be ready
    print_info "Waiting for backend to be ready..."
    local count=0
    while [ $count -lt 30 ]; do
        if curl -s "http://localhost:8000/api/health" > /dev/null 2>&1; then
            print_success "Backend is ready"
            break
        fi
        sleep 1
        count=$((count + 1))
    done
    
    if [ $count -eq 30 ]; then
        print_error "Backend failed to start within 30 seconds"
        return 1
    fi
    
    # Start frontend
    print_info "Starting frontend service..."
    if [ "$background" = true ]; then
        "$FRONTEND_SCRIPT" start -b
    else
        # For foreground mode, start frontend in background since we want to keep the script running
        "$FRONTEND_SCRIPT" start -b
    fi
    
    print_success "Development environment started successfully!"
    print_info "Backend API: http://localhost:8000"
    print_info "Web UI: http://localhost:8000"
    print_info "Frontend Assets: http://localhost:3000"
    
    if [ "$background" = false ]; then
        print_info "Press Ctrl+C to stop all services"
        
        # Wait for interrupt signal
        trap 'stop_all; exit 0' INT
        while true; do
            sleep 1
        done
    fi
}

# Function to stop all services
stop_all() {
    print_status "Stopping MCP Training Service development environment..."
    
    # Stop frontend first
    print_info "Stopping frontend service..."
    "$FRONTEND_SCRIPT" stop
    
    # Stop backend
    print_info "Stopping backend service..."
    "$BACKEND_SCRIPT" stop
    
    print_success "All services stopped"
}

# Function to restart all services
restart_all() {
    local background=${1:-false}
    
    print_status "Restarting MCP Training Service development environment..."
    stop_all
    sleep 2
    start_all "$background"
}

# Function to show status of all services
show_status() {
    print_status "MCP Training Service Development Environment Status"
    echo ""
    
    local backend_status=$(check_service_status "mcp-training-backend")
    local frontend_status=$(check_service_status "mcp-training-frontend")
    
    echo "Backend Service:  $backend_status"
    echo "Frontend Service: $frontend_status"
    echo ""
    
    # Check if services are accessible
    if echo "$backend_status" | grep -q "running"; then
        if curl -s "http://localhost:8000/api/health" > /dev/null 2>&1; then
            print_success "Backend API is accessible at http://localhost:8000"
        else
            print_warning "Backend is running but API is not responding"
        fi
    fi
    
    if echo "$frontend_status" | grep -q "running"; then
        if curl -s "http://localhost:3000" > /dev/null 2>&1; then
            print_success "Frontend is accessible at http://localhost:3000"
        else
            print_warning "Frontend is running but not responding"
        fi
    fi
}

# Function to show logs
show_logs() {
    local service="$1"
    
    case "$service" in
        backend|b)
            print_info "Showing backend logs..."
            "$BACKEND_SCRIPT" logs
            ;;
        frontend|f)
            print_info "Showing frontend logs..."
            "$FRONTEND_SCRIPT" logs
            ;;
        all)
            print_info "Showing all logs (backend first, then frontend)..."
            echo "=== BACKEND LOGS ==="
            "$BACKEND_SCRIPT" logs &
            local backend_pid=$!
            sleep 2
            echo "=== FRONTEND LOGS ==="
            "$FRONTEND_SCRIPT" logs &
            local frontend_pid=$!
            
            # Wait for both processes
            wait $backend_pid $frontend_pid
            ;;
        *)
            print_error "Unknown service: $service"
            print_info "Available services: backend, frontend, all"
            exit 1
            ;;
    esac
}

# Function to setup development environment
setup_dev() {
    print_status "Setting up development environment..."
    
    # Check if virtual environment exists
    if [ ! -d "$PROJECT_ROOT/venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv "$PROJECT_ROOT/venv"
    fi
    
    # Activate virtual environment and install dependencies
    print_info "Installing dependencies..."
    source "$PROJECT_ROOT/venv/bin/activate"
    
    # Upgrade pip and setuptools first (fixes Raspberry Pi issues)
    print_info "Upgrading pip and setuptools..."
    pip install --upgrade pip setuptools wheel
    
    # Install numpy first (it's a dependency for other packages)
    print_info "Installing numpy first..."
    pip install numpy==1.24.3
    
    # Install the rest of the requirements
    print_info "Installing remaining dependencies..."
    pip install -r "$PROJECT_ROOT/requirements.txt"
    
    # Create necessary directories
    print_info "Creating necessary directories..."
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/models"
    mkdir -p "$PROJECT_ROOT/exports"
    
    print_success "Development environment setup complete!"
}

# Function to clean up
cleanup() {
    print_status "Cleaning up development environment..."
    
    # Stop all services
    stop_all
    
    # Remove PID files
    rm -f "$PROJECT_ROOT/logs/mcp-training-backend.pid"
    rm -f "$PROJECT_ROOT/logs/mcp-training-frontend.pid"
    
    # Run comprehensive cleanup of generated data
    print_info "Running comprehensive data cleanup..."
    "$SCRIPT_DIR/cleanup.sh" --confirm
    
    print_success "Cleanup complete!"
}

# Function to show help
show_help() {
    echo "MCP Training Service Development Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start     Start all development services"
    echo "  stop      Stop all development services"
    echo "  restart   Restart all development services"
    echo "  status    Show status of all services"
    echo "  logs      Show service logs"
    echo "  setup     Setup development environment"
    echo "  cleanup   Clean up development environment (stops services and removes generated data)"
    echo "  help      Show this help message"
    echo ""
    echo "Options:"
    echo "  -b, --background  Start services in background mode"
    echo "  -f, --foreground  Start services in foreground mode (default)"
    echo ""
    echo "Log Commands:"
    echo "  $0 logs backend    # Show backend logs"
    echo "  $0 logs frontend   # Show frontend logs"
    echo "  $0 logs all        # Show all logs"
    echo ""
    echo "Examples:"
    echo "  $0 setup           # Setup development environment"
    echo "  $0 start           # Start all services in foreground"
    echo "  $0 start -b        # Start all services in background"
    echo "  $0 stop            # Stop all services"
    echo "  $0 restart         # Restart all services"
    echo "  $0 status          # Show service status"
    echo "  $0 logs backend    # Show backend logs"
    echo "  $0 cleanup         # Clean up environment"
    echo ""
    echo "Services:"
    echo "  Backend API:  http://localhost:8000"
    echo "  Web UI:       http://localhost:8000"
    echo "  Frontend:     http://localhost:3000"
}

# Main script logic
case "${1:-help}" in
    start)
        background=false
        if [ "$2" = "-b" ] || [ "$2" = "--background" ]; then
            background=true
        fi
        start_all "$background"
        ;;
    stop)
        stop_all
        ;;
    restart)
        background=false
        if [ "$2" = "-b" ] || [ "$2" = "--background" ]; then
            background=true
        fi
        restart_all "$background"
        ;;
    status)
        show_status
        ;;
    logs)
        service="${2:-all}"
        show_logs "$service"
        ;;
    setup)
        setup_dev
        ;;
    cleanup)
        cleanup
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