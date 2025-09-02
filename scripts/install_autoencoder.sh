#!/bin/bash

# HEP-ML-Templates Autoencoder Installation Script
# This script installs the necessary dependencies for Autoencoder pipelines

echo "🔧 Installing HEP-ML-Templates Autoencoder Components..."

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

echo "📦 Installing Matplotlib..."
pip3 install matplotlib

echo "📦 Installing HEP-ML-Templates with Autoencoder support..."
# Install the package in editable mode with Autoencoder extras
pip3 install -e ".[pipeline-autoencoder]"

echo "🧪 Testing Autoencoder installation..."
python3 -c "import torch; import lightning; import matplotlib; print('✅ Autoencoder components installed successfully!')"

if [ $? -eq 0 ]; then
    echo "🎉 Autoencoder installation completed successfully!"
echo "🚀 You can now run Autoencoder pipelines by configuring your pipeline.yaml file"
else
    echo "❌ Autoencoder installation failed. Please check the error messages above."
    exit 1
fi
