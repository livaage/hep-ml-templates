#!/bin/bash

# HEP-ML-Templates Complete Installation Script
# This script installs ALL dependencies for ALL pipelines

echo "🔧 Installing HEP-ML-Templates Complete Package..."

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

echo "📦 Installing PyTorch Geometric..."
pip3 install torch-geometric

echo "📦 Installing XGBoost..."
pip3 install xgboost

echo "📦 Installing Uproot..."
pip3 install uproot

echo "📦 Installing Awkward..."
pip3 install awkward

echo "📦 Installing Matplotlib..."
pip3 install matplotlib

echo "📦 Installing HEP-ML-Templates with ALL support..."
# Install the package in editable mode with ALL extras
pip3 install -e ".[all]"

echo "🧪 Testing complete installation..."
python3 -c "
import torch
import lightning
import torch_geometric
import xgboost
import uproot
import awkward
import matplotlib
print('✅ All components installed successfully!')
"

if [ $? -eq 0 ]; then
    echo "🎉 Complete installation finished successfully!"
echo "🚀 You can now run ALL pipeline types by configuring your pipeline.yaml file"
    echo ""
    echo "Available pipelines:"
    echo "  - XGBoost"
    echo "  - Decision Tree"
    echo "  - Ensemble"
    echo "  - PyTorch Neural Networks"
    echo "  - Neural Network (MLP)"
    echo "  - Graph Neural Networks (GNN)"
    echo "  - Autoencoders"
    echo "  - And more!"
else
    echo "❌ Complete installation failed. Please check the error messages above."
    exit 1
fi
