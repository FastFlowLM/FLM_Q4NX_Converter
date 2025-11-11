#!/bin/bash
# =============================================================================
# File        : setup_venv.sh
# Author      : Zhenyu Xu, zxu3@clemson.edu
# Created     : 2025-03-27
# Description : This is a script to setup the virtual environment for the Quark project.
# =============================================================================

# Set environment name and Python version (if needed)
VENV_DIR="venv"
PYTHON_CMD="python3"  # Change to python3.x if necessary

# Check if Python is installed
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Error: $PYTHON_CMD is not installed. Please install Python and try again."
    exit 1
fi

# Check if virtual environment already exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists in $VENV_DIR. Skipping creation."
    source $VENV_DIR/bin/activate
else
    # Create virtual environment
    echo "Creating virtual environment in $VENV_DIR..."
    $PYTHON_CMD -m venv $VENV_DIR
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Upgrade
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies if requirements.txt exists
if [[ -f "requirements.txt" ]]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Skipping dependency installation."
    pip install torch
    pip install einops
    pip install matplotlib
    pip install ipykernel
    pip install amd-quark
    pip install accelerate
    pip install "huggingface-hub[cli]"
    pip install gguf
    pip install git+https://github.com/huggingface/transformers.git
fi

# Print success message
echo "Virtual environment setup complete. Activate it using:"
echo "source $VENV_DIR/bin/activate"

mkdir ../gguf_files
mkdir ../q4nx_files
