#!/bin/bash

# HEP-ML-Templates Neural Network Installation Script
# This script installs the necessary dependencies for Neural Network (MLP) pipelines

echo "🔧 Installing HEP-ML-Templates Neural Network Components..."

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

echo "📦 Installing HEP-ML-Templates with Neural Network support..."
# Install the package in editable mode with Neural Network extras
pip3 install -e ".[pipeline-neural]"

echo "🧪 Testing Neural Network installation..."
python3 -c "from sklearn.neural_network import MLPClassifier; print('✅ Neural Network components installed successfully!')"

if [ $? -eq 0 ]; then
    echo "🎉 Neural Network installation completed successfully!"
echo "🚀 You can now run Neural Network pipelines by configuring your pipeline.yaml file"
else
    echo "❌ Neural Network installation failed. Please check the error messages above."
    exit 1
fi
