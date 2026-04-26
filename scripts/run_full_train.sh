#!/usr/bin/env bash
set -e

echo "=== Running EPGNN Full Scale Training ==="

# Check if real data exists
if [ ! -f "stead_earthquake.csv" ]; then
    echo "Real STEAD data not found. Initiating download..."
    bash scripts/download_datasets.sh
fi

# Run R data prep
echo "Running R data preprocessing..."
Rscript data/data_prep.R

# Train the model
echo "Training model on target GPU..."
python main.py --mode train --epochs 50

# Evaluate the model
echo "Evaluating model..."
python main.py --mode evaluate

echo "=== Full Scale Training Complete ==="
