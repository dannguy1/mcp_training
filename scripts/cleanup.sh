#!/bin/bash

# MCP Training Service Cleanup Script
# This script removes generated data files while preserving essential structure

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

# Function to confirm action
confirm_action() {
    local message="$1"
    echo -e "${YELLOW}$message${NC}"
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cleanup cancelled by user"
        exit 0
    fi
}

# Function to cleanup exports directory
cleanup_exports() {
    print_status "Cleaning up exports directory..."
    
    local exports_dir="$PROJECT_ROOT/exports"
    if [ -d "$exports_dir" ]; then
        # Remove all export files except sample_export.json
        find "$exports_dir" -name "*.json" ! -name "sample_export.json" -type f -delete
        print_success "Removed generated export files (preserved sample_export.json)"
    else
        print_warning "Exports directory not found"
    fi
}

# Function to cleanup models directory
cleanup_models() {
    print_status "Cleaning up models directory..."
    
    local models_dir="$PROJECT_ROOT/models"
    if [ -d "$models_dir" ]; then
        # Remove all model deployment files
        if [ -d "$models_dir/deployments" ]; then
            rm -rf "$models_dir/deployments"/*
            print_success "Removed model deployment files"
        fi
        
        # Remove model registry (it will be regenerated)
        if [ -f "$models_dir/model_registry.json" ]; then
            rm "$models_dir/model_registry.json"
            print_success "Removed model registry (will be regenerated)"
        fi
    else
        print_warning "Models directory not found"
    fi
}

# Function to cleanup logs directory
cleanup_logs() {
    print_status "Cleaning up logs directory..."
    
    local logs_dir="$PROJECT_ROOT/logs"
    if [ -d "$logs_dir" ]; then
        # Remove all log files
        find "$logs_dir" -name "*.log*" -type f -delete
        print_success "Removed log files"
        
        # Remove PID files
        find "$logs_dir" -name "*.pid" -type f -delete
        print_success "Removed PID files"
    else
        print_warning "Logs directory not found"
    fi
}

# Function to cleanup cache directories
cleanup_cache() {
    print_status "Cleaning up cache directories..."
    
    # Remove Python cache
    if [ -d "$PROJECT_ROOT/__pycache__" ]; then
        rm -rf "$PROJECT_ROOT/__pycache__"
        print_success "Removed Python cache"
    fi
    
    # Remove pytest cache
    if [ -d "$PROJECT_ROOT/.pytest_cache" ]; then
        rm -rf "$PROJECT_ROOT/.pytest_cache"
        print_success "Removed pytest cache"
    fi
    
    # Remove benchmark cache
    if [ -d "$PROJECT_ROOT/.benchmarks" ]; then
        rm -rf "$PROJECT_ROOT/.benchmarks"
        print_success "Removed benchmark cache"
    fi
    
    # Remove Python cache in src directory
    if [ -d "$PROJECT_ROOT/src/__pycache__" ]; then
        rm -rf "$PROJECT_ROOT/src/__pycache__"
        print_success "Removed src Python cache"
    fi
    
    # Remove Python cache in tests directory
    if [ -d "$PROJECT_ROOT/tests/__pycache__" ]; then
        rm -rf "$PROJECT_ROOT/tests/__pycache__"
        print_success "Removed tests Python cache"
    fi
}

# Function to cleanup temporary files
cleanup_temp() {
    print_status "Cleaning up temporary files..."
    
    # Remove any temporary files in the project root
    find "$PROJECT_ROOT" -name "*.tmp" -type f -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.temp" -type f -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*~" -type f -delete 2>/dev/null || true
    
    print_success "Removed temporary files"
}

# Function to show cleanup summary
show_summary() {
    print_status "Cleanup Summary"
    echo ""
    echo "Directories cleaned:"
    echo "  ✓ exports/ - Removed generated export files"
    echo "  ✓ models/ - Removed model deployments and registry"
    echo "  ✓ logs/ - Removed log files and PID files"
    echo "  ✓ __pycache__ - Removed Python cache"
    echo "  ✓ .pytest_cache - Removed pytest cache"
    echo "  ✓ .benchmarks - Removed benchmark cache"
    echo ""
    echo "Preserved files:"
    echo "  ✓ sample_export.json - Essential sample file"
    echo "  ✓ Directory structure - All directories maintained"
    echo ""
    print_success "Cleanup completed successfully!"
}

# Function to show help
show_help() {
    echo "MCP Training Service Cleanup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "This script removes generated data files while preserving essential structure."
    echo ""
    echo "Options:"
    echo "  --confirm    Skip confirmation prompt"
    echo "  --dry-run    Show what would be cleaned without actually doing it"
    echo "  --help       Show this help message"
    echo ""
    echo "What gets cleaned:"
    echo "  - Generated export files (preserves sample_export.json)"
    echo "  - Model deployment files and registry"
    echo "  - Log files and PID files"
    echo "  - Python cache files (__pycache__)"
    echo "  - Pytest cache (.pytest_cache)"
    echo "  - Benchmark cache (.benchmarks)"
    echo "  - Temporary files"
    echo ""
    echo "What gets preserved:"
    echo "  - Directory structure"
    echo "  - sample_export.json"
    echo "  - Configuration files"
    echo "  - Source code"
    echo ""
    echo "Example:"
    echo "  $0 --confirm    # Clean without confirmation"
    echo "  $0 --dry-run    # Show what would be cleaned"
}

# Function to perform dry run
dry_run() {
    print_status "DRY RUN - Showing what would be cleaned..."
    echo ""
    
    local exports_dir="$PROJECT_ROOT/exports"
    if [ -d "$exports_dir" ]; then
        echo "Exports to remove:"
        find "$exports_dir" -name "*.json" ! -name "sample_export.json" -type f -printf "  %p\n" 2>/dev/null || true
        echo ""
    fi
    
    local models_dir="$PROJECT_ROOT/models"
    if [ -d "$models_dir" ]; then
        echo "Models to remove:"
        if [ -d "$models_dir/deployments" ]; then
            find "$models_dir/deployments" -type f -printf "  %p\n" 2>/dev/null || true
        fi
        if [ -f "$models_dir/model_registry.json" ]; then
            echo "  $models_dir/model_registry.json"
        fi
        echo ""
    fi
    
    local logs_dir="$PROJECT_ROOT/logs"
    if [ -d "$logs_dir" ]; then
        echo "Logs to remove:"
        find "$logs_dir" -name "*.log*" -type f -printf "  %p\n" 2>/dev/null || true
        find "$logs_dir" -name "*.pid" -type f -printf "  %p\n" 2>/dev/null || true
        echo ""
    fi
    
    echo "Cache directories to remove:"
    [ -d "$PROJECT_ROOT/__pycache__" ] && echo "  $PROJECT_ROOT/__pycache__"
    [ -d "$PROJECT_ROOT/.pytest_cache" ] && echo "  $PROJECT_ROOT/.pytest_cache"
    [ -d "$PROJECT_ROOT/.benchmarks" ] && echo "  $PROJECT_ROOT/.benchmarks"
    [ -d "$PROJECT_ROOT/src/__pycache__" ] && echo "  $PROJECT_ROOT/src/__pycache__"
    [ -d "$PROJECT_ROOT/tests/__pycache__" ] && echo "  $PROJECT_ROOT/tests/__pycache__"
    echo ""
    
    print_info "Dry run completed. No files were actually removed."
}

# Main script logic
main() {
    local confirm=false
    local dry_run=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --confirm)
                confirm=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    print_status "Starting MCP Training Service cleanup..."
    echo ""
    
    if [ "$dry_run" = true ]; then
        dry_run
        exit 0
    fi
    
    if [ "$confirm" = false ]; then
        confirm_action "This will remove all generated data files including exports, models, logs, and cache files."
    fi
    
    # Perform cleanup
    cleanup_exports
    cleanup_models
    cleanup_logs
    cleanup_cache
    cleanup_temp
    
    echo ""
    show_summary
}

# Run main function
main "$@" 