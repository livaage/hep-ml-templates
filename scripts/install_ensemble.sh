#!/bin/bash

# HEP-ML-Templates Ensemble Installation Script
# This script installs the necessary dependencies for Ensemble pipelines

echo "🔧 Installing HEP-ML-Templates Ensemble Components..."

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

echo "📦 Installing HEP-ML-Templates with Ensemble support..."
# Install the package in editable mode with Ensemble extras
pip3 install -e ".[pipeline-ensemble]"

echo "🧪 Testing Ensemble installation..."
python3 -c "import xgboost; from sklearn.ensemble import VotingClassifier; print('✅ Ensemble components installed successfully!')"

if [ $? -eq 0 ]; then
    echo "🎉 Ensemble installation completed successfully!"
echo "🚀 You can now run Ensemble pipelines by configuring your pipeline.yaml file"
else
    echo "❌ Ensemble installation failed. Please check the error messages above."
    exit 1
fi
