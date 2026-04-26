#!/usr/bin/env bash
set -e

echo "=== Downloading STEAD Dataset from Kaggle ==="

# Check if kaggle API is installed
if ! command -v kaggle &> /dev/null; then
    echo "Kaggle CLI not found. Installing..."
    pip install kaggle
fi

# Note: Requires ~/.kaggle/kaggle.json with API credentials
echo "Starting download (This is ~80GB, please be patient)..."
kaggle datasets download -d mostafa/stead --unzip

echo "=== Download Complete ==="
