#!/usr/bin/env bash
set -e

echo "=== Running EPGNN Smoke Test ==="

# 1. Generate synthetic mock data
python main.py --mode mock

# 2. Run R data prep
echo "Running R data preprocessing..."
Rscript epgnn/data/data_prep.R

# 3. Train and Evaluate
python main.py --mode train --epochs 2
python main.py --mode evaluate

echo "=== Smoke Test Complete ==="
