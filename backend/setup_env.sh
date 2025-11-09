#!/bin/bash
# Setup script for backend Python environment
# This script creates a virtual environment with the correct Python version
# and installs all dependencies.

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
PYTHON_MIN_VERSION="3.11"
PYTHON_MAX_VERSION="3.13"  # PyTorch doesn't support 3.14 yet

echo "Setting up backend Python environment..."

# Function to check Python version
check_python_version() {
    local python_cmd=$1
    if command -v "$python_cmd" &> /dev/null; then
        local version=$($python_cmd --version 2>&1 | awk '{print $2}')
        local major=$(echo "$version" | cut -d. -f1)
        local minor=$(echo "$version" | cut -d. -f2)
        local major_min=$(echo "$PYTHON_MIN_VERSION" | cut -d. -f1)
        local minor_min=$(echo "$PYTHON_MIN_VERSION" | cut -d. -f2)
        local major_max=$(echo "$PYTHON_MAX_VERSION" | cut -d. -f1)
        local minor_max=$(echo "$PYTHON_MAX_VERSION" | cut -d. -f2)
        
        if [ "$major" -eq "$major_min" ] && [ "$minor" -ge "$minor_min" ]; then
            if [ "$major" -lt "$major_max" ] || ([ "$major" -eq "$major_max" ] && [ "$minor" -le "$minor_max" ]); then
                echo "$python_cmd"
                return 0
            fi
        fi
    fi
    return 1
}

# Find suitable Python version
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3; do
    if check_python_version "$cmd"; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: No suitable Python version found!"
    echo "Required: Python $PYTHON_MIN_VERSION to $PYTHON_MAX_VERSION"
    echo ""
    echo "Please install Python 3.11 or 3.12:"
    echo "  macOS: brew install python@3.12"
    echo "  Ubuntu/Debian: sudo apt-get install python3.12 python3.12-venv"
    echo "  Or download from: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version)
echo "Using: $PYTHON_VERSION ($PYTHON_CMD)"

# Remove existing venv if it exists and was created with wrong Python version
if [ -d "$VENV_DIR" ]; then
    if [ -f "$VENV_DIR/pyvenv.cfg" ]; then
        VENV_PYTHON=$(grep "version" "$VENV_DIR/pyvenv.cfg" | cut -d'=' -f2 | tr -d ' ')
        CURRENT_PYTHON=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
        
        if [ "$VENV_PYTHON" != "$CURRENT_PYTHON" ]; then
            echo "Existing venv uses Python $VENV_PYTHON, but we need $CURRENT_PYTHON"
            echo "Removing existing venv..."
            rm -rf "$VENV_DIR"
        fi
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    echo "Virtual environment created at $VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install dependencies
echo "Installing dependencies..."
pip install -e . > /dev/null 2>&1

echo ""
echo "✓ Backend environment setup complete!"
echo ""
echo "To activate the environment manually, run:"
echo "  source backend/venv/bin/activate"
echo ""
echo "Or use the Makefile commands:"
echo "  make setup-backend  # Re-run this setup"
echo "  make test           # Run tests"

