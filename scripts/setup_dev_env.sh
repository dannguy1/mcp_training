#!/bin/bash

# Raspberry Pi Development Environment Setup Script
# This script handles the specific issues with Python 3.12 on Raspberry Pi

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

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

# Function to check system requirements
check_system() {
    print_status "Checking system requirements..."
    
    # Check Python version
    local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_info "Python version: $python_version"
    
    # Check if we're on Raspberry Pi
    if [ -f "/proc/cpuinfo" ] && grep -q "Raspberry Pi" /proc/cpuinfo; then
        print_info "Detected Raspberry Pi system"
        RPI_MODE=true
    else
        print_info "Non-Raspberry Pi system detected"
        RPI_MODE=false
    fi
    
    # Check available memory
    local mem_total=$(free -m | awk 'NR==2{printf "%.0f", $2}')
    print_info "Total memory: ${mem_total}MB"
    
    if [ "$mem_total" -lt 2048 ]; then
        print_warning "Low memory detected. Some packages may take longer to install."
    fi
}

# Function to setup virtual environment
setup_venv() {
    print_status "Setting up virtual environment..."
    
    # Remove existing venv if it exists
    if [ -d "$PROJECT_ROOT/venv" ]; then
        print_info "Removing existing virtual environment..."
        rm -rf "$PROJECT_ROOT/venv"
    fi
    
    # Create new virtual environment
    print_info "Creating new virtual environment..."
    python3 -m venv "$PROJECT_ROOT/venv"
    
    # Activate virtual environment
    source "$PROJECT_ROOT/venv/bin/activate"
    
    print_success "Virtual environment created and activated"
}

# Function to install system dependencies (Raspberry Pi specific)
install_system_deps() {
    if [ "$RPI_MODE" = true ]; then
        print_status "Installing system dependencies for Raspberry Pi..."
        
        # Update package list
        sudo apt-get update
        
        # Install build dependencies
        sudo apt-get install -y \
            build-essential \
            python3-dev \
            libatlas-base-dev \
            gfortran \
            libopenblas-dev \
            liblapack-dev \
            pkg-config
        
        print_success "System dependencies installed"
    else
        print_info "Skipping system dependencies (not on Raspberry Pi)"
    fi
}

# Function to install Python packages
install_python_packages() {
    print_status "Installing Python packages..."
    
    # Activate virtual environment
    source "$PROJECT_ROOT/venv/bin/activate"
    
    # Upgrade pip and setuptools first
    print_info "Upgrading pip and setuptools..."
    pip install --upgrade pip setuptools wheel
    
    # Install packages in specific order for Raspberry Pi compatibility
    if [ "$RPI_MODE" = true ]; then
        print_info "Installing packages with Raspberry Pi optimizations..."
        
        # Install numpy first (with specific flags for RPi)
        print_info "Installing numpy..."
        pip install numpy --no-cache-dir
        
        # Install scipy (dependency for scikit-learn)
        print_info "Installing scipy..."
        pip install scipy --no-cache-dir
        
        # Install packages one by one to avoid conflicts
        print_info "Installing FastAPI and web framework..."
        pip install fastapi==0.104.1 uvicorn[standard]==0.24.0 --no-cache-dir
        
        print_info "Installing Pydantic..."
        pip install pydantic==2.5.0 pydantic-settings==2.1.0 --no-cache-dir
        
        print_info "Installing data processing packages..."
        pip install pandas scikit-learn joblib --no-cache-dir
        
        print_info "Installing templating packages..."
        pip install jinja2 markupsafe --no-cache-dir
        
        print_info "Installing utility packages..."
        pip install click==8.1.7 pyyaml==6.0.1 python-multipart==0.0.6 aiofiles==23.2.1 --no-cache-dir
        
        print_info "Installing monitoring packages..."
        pip install prometheus-client==0.19.0 psutil --no-cache-dir
        
        print_info "Installing remaining packages..."
        pip install asyncio-mqtt==0.16.1 httpx==0.25.2 pathlib2==2.3.7 python-dateutil==2.8.2 --no-cache-dir
    else
        # Standard installation for non-RPi systems
        print_info "Installing packages..."
        pip install -r "$PROJECT_ROOT/requirements.txt"
    fi
    
    print_success "Python packages installed"
}

# Function to create directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/models"
    mkdir -p "$PROJECT_ROOT/exports"
    mkdir -p "$PROJECT_ROOT/config"
    
    print_success "Directories created"
}

# Function to verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Activate virtual environment
    source "$PROJECT_ROOT/venv/bin/activate"
    
    # Test key imports
    print_info "Testing key package imports..."
    python3 -c "
import sys
print('Python version:', sys.version)

try:
    import fastapi
    print('✓ FastAPI imported successfully')
except ImportError as e:
    print('✗ FastAPI import failed:', e)
    sys.exit(1)

try:
    import numpy
    print('✓ NumPy imported successfully')
except ImportError as e:
    print('✗ NumPy import failed:', e)
    sys.exit(1)

try:
    import pandas
    print('✓ Pandas imported successfully')
except ImportError as e:
    print('✗ Pandas import failed:', e)
    sys.exit(1)

try:
    import sklearn
    print('✓ Scikit-learn imported successfully')
except ImportError as e:
    print('✗ Scikit-learn import failed:', e)
    sys.exit(1)

print('✓ All key packages imported successfully')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation verification passed"
    else
        print_error "Installation verification failed"
        exit 1
    fi
}

# Function to show next steps
show_next_steps() {
    print_success "Development environment setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment:"
    echo "   source $PROJECT_ROOT/venv/bin/activate"
    echo ""
    echo "2. Start the development services:"
    echo "   ./scripts/dev.sh start"
    echo ""
    echo "3. Access the application:"
    echo "   Web UI: http://localhost:8000"
    echo "   API: http://localhost:8000/api"
    echo ""
    echo "4. View logs:"
    echo "   ./scripts/dev.sh logs"
    echo ""
    echo "5. Stop services:"
    echo "   ./scripts/dev.sh stop"
}

# Main execution
main() {
    print_status "Starting MCP Training Service development environment setup..."
    
    check_system
    setup_venv
    install_system_deps
    install_python_packages
    create_directories
    verify_installation
    show_next_steps
}

# Run main function
main "$@" 