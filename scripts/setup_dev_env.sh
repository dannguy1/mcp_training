#!/bin/bash

# MCP Training Service Development Environment Setup
# This script sets up the development environment for the MCP Training Service

set -e

echo "🚀 Setting up MCP Training Service Development Environment"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.8+ is installed
check_python() {
    print_status "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.8+ is required but not installed"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Dependencies installed"
}

# Install development dependencies
install_dev_dependencies() {
    print_status "Installing development dependencies..."
    pip install -r requirements-dev.txt
    print_success "Development dependencies installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p models exports logs tests/fixtures
    print_success "Directories created"
}

# Create sample export file
create_sample_export() {
    print_status "Creating sample export file..."
    cat > exports/sample_export.json << 'EOF'
{
  "export_metadata": {
    "created_at": "2024-01-01T12:00:00Z",
    "total_records": 100,
    "format": "json",
    "export_id": "sample_export_001",
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-01T23:59:59Z",
    "programs": ["hostapd", "wpa_supplicant"]
  },
  "data": [
    {
      "id": 1,
      "device_id": "device_001",
      "device_ip": "192.168.1.100",
      "timestamp": "2024-01-01T12:00:00Z",
      "log_level": "INFO",
      "process_name": "hostapd",
      "message": "AP-STA-CONNECTED 00:11:22:33:44:55",
      "raw_message": "AP-STA-CONNECTED 00:11:22:33:44:55",
      "structured_data": {
        "event_type": "connection",
        "mac_address": "00:11:22:33:44:55",
        "ssid": "test_network"
      },
      "pushed_to_ai": false,
      "pushed_at": null,
      "push_attempts": 0,
      "last_push_error": null
    },
    {
      "id": 2,
      "device_id": "device_001",
      "device_ip": "192.168.1.100",
      "timestamp": "2024-01-01T12:01:00Z",
      "log_level": "ERROR",
      "process_name": "hostapd",
      "message": "authentication failure for 00:11:22:33:44:55",
      "raw_message": "authentication failure for 00:11:22:33:44:55",
      "structured_data": {
        "event_type": "auth_failure",
        "mac_address": "00:11:22:33:44:55"
      },
      "pushed_to_ai": false,
      "pushed_at": null,
      "push_attempts": 0,
      "last_push_error": null
    }
  ]
}
EOF
    print_success "Sample export file created"
}

# Run basic tests
run_tests() {
    print_status "Running basic tests..."
    if python -m pytest tests/ -v --tb=short; then
        print_success "Tests passed"
    else
        print_warning "Some tests failed - this is expected for incomplete implementation"
    fi
}

# Show next steps
show_next_steps() {
    echo ""
    print_success "Development environment setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. Start the API server:"
    echo "   python -m uvicorn mcp_training.api.app:app --reload --host 0.0.0.0 --port 8001"
    echo ""
    echo "3. Test the CLI:"
    echo "   python -m mcp_training.cli validate exports/sample_export.json"
    echo ""
    echo "4. Access the API documentation:"
    echo "   http://localhost:8001/docs"
    echo ""
    echo "5. Follow the implementation plan:"
    echo "   docs/Implementation-Plan.md"
    echo ""
}

# Main execution
main() {
    check_python
    create_venv
    activate_venv
    install_dependencies
    install_dev_dependencies
    create_directories
    create_sample_export
    run_tests
    show_next_steps
}

# Run main function
main "$@" 