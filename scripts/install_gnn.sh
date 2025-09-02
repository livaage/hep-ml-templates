#!/bin/bash

# HEP-ML-Templates GNN Installation Script
# This script installs the necessary dependencies for Graph Neural Networks

echo "🔧 Installing HEP-ML-Templates GNN Components..."

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

echo "📦 Installing PyTorch Geometric..."
pip3 install torch-geometric

echo "📦 Installing HEP-ML-Templates with GNN support..."
# Install the package in editable mode with GNN extras
pip3 install -e ".[pipeline-gnn]"

echo "🧪 Testing GNN installation..."
python3 -c "import torch_geometric; print('✅ PyTorch Geometric installed successfully!')"

if [ $? -eq 0 ]; then
    echo "🎉 GNN installation completed successfully!"
echo "🚀 You can now run GNN pipelines by configuring your pipeline.yaml file"
else
    echo "❌ GNN installation failed. Please check the error messages above."
    exit 1
fi
