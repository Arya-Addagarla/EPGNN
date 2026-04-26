#!/usr/bin/env bash
set -e

echo "=== Running EPGNN Full Scale Training ==="

# Check if real data exists
if [ ! -f "chunk2.hdf5" ]; then
    echo "Error: Real STEAD data (e.g., chunk2.hdf5) not found in the root directory."
    echo "Please download the STEAD dataset from Kaggle before running this."
    exit 1
fi

# Run R data prep
echo "Running R data preprocessing..."
Rscript epgnn/data/data_prep.R

# Train the model
echo "Training model on target GPU..."
python main.py --mode train --epochs 50

# Evaluate the model
echo "Evaluating model..."
python main.py --mode evaluate

echo "=== Full Scale Training Complete ==="
