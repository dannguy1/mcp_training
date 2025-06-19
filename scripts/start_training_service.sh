#!/bin/bash

# MCP Training Service Startup Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the mcp_training directory"
    exit 1
fi

print_status "Starting MCP Training Service..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_error "Python 3.10 or higher is required. Found: $python_version"
    exit 1
fi

print_status "Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
print_status "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
print_status "Creating directories..."
mkdir -p models exports logs

# Set up environment
if [ ! -f ".env" ]; then
    print_status "Creating .env file from template..."
    cp env.example .env
    print_warning "Please review and edit .env file with your settings"
fi

# Check if API server should be started
if [ "$1" = "--api" ] || [ "$1" = "-a" ]; then
    print_status "Starting API server..."
    export PYTHONPATH=$PWD/src
    uvicorn mcp_training.api.app:app --host 0.0.0.0 --port 8001 --reload
else
    print_status "Setup complete!"
    echo ""
    echo "Available commands:"
    echo "  python -m mcp_training.cli train <export_file>    # Train a model"
    echo "  python -m mcp_training.cli validate <export_file> # Validate export data"
    echo "  python -m mcp_training.cli list-models           # List trained models"
    echo "  python -m mcp_training.cli info                  # Show system info"
    echo ""
    echo "To start the API server, run:"
    echo "  ./scripts/start_training_service.sh --api"
    echo ""
    echo "Or manually:"
    echo "  uvicorn mcp_training.api.app:app --host 0.0.0.0 --port 8001"
fi 