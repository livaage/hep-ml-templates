#!/bin/bash

# HEP-ML-Templates XGBoost Installation Script
# This script installs the necessary dependencies for XGBoost pipelines

echo "🔧 Installing HEP-ML-Templates XGBoost Components..."

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

echo "📦 Installing XGBoost..."
pip3 install xgboost

echo "📦 Installing HEP-ML-Templates with XGBoost support..."
# Install the package in editable mode with XGBoost extras
pip3 install -e ".[pipeline-xgb]"

echo "🧪 Testing XGBoost installation..."
python3 -c "import xgboost; print('✅ XGBoost installed successfully!')"

if [ $? -eq 0 ]; then
    echo "🎉 XGBoost installation completed successfully!"
echo "🚀 You can now run XGBoost pipelines by configuring your pipeline.yaml file"
else
    echo "❌ XGBoost installation failed. Please check the error messages above."
    exit 1
fi
