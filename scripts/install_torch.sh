#!/bin/bash

# HEP-ML-Templates PyTorch Installation Script
# This script installs the necessary dependencies for PyTorch pipelines

echo "🔧 Installing HEP-ML-Templates PyTorch Components..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "📦 Installing PyTorch..."
pip3 install torch

echo "📦 Installing PyTorch Lightning..."
pip3 install lightning

echo "📦 Installing HEP-ML-Templates with PyTorch support..."
# Install the package in editable mode with PyTorch extras
pip3 install -e ".[pipeline-torch]"

echo "🧪 Testing PyTorch installation..."
python3 -c "import torch; import lightning; print('✅ PyTorch and Lightning installed successfully!')"

if [ $? -eq 0 ]; then
    echo "🎉 PyTorch installation completed successfully!"
echo "🚀 You can now run PyTorch pipelines by configuring your pipeline.yaml file"
else
    echo "❌ PyTorch installation failed. Please check the error messages above."
    exit 1
fi
