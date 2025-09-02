#!/bin/bash

# HEP-ML-Templates Decision Tree Installation Script
# This script installs the necessary dependencies for Decision Tree pipelines

echo "🔧 Installing HEP-ML-Templates Decision Tree Components..."

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

echo "📦 Installing HEP-ML-Templates with Decision Tree support..."
# Install the package in editable mode with Decision Tree extras
pip3 install -e ".[pipeline-decision-tree]"

echo "🧪 Testing Decision Tree installation..."
python3 -c "from sklearn.tree import DecisionTreeClassifier; print('✅ Decision Tree components installed successfully!')"

if [ $? -eq 0 ]; then
    echo "🎉 Decision Tree installation completed successfully!"
echo "🚀 You can now run Decision Tree pipelines by configuring your pipeline.yaml file"
else
    echo "❌ Decision Tree installation failed. Please check the error messages above."
    exit 1
fi
