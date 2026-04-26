# EPGNN (Earthquake Prediction Graph Neural Network)

This repository contains the PyTorch-based implementation of EPGNN, a research framework designed to detect early-warning seismic events and learn robust representations from multivariate seismogram waveforms. 

The architecture leverages Multimodal Graph Neural Networks to extract both spatial and temporal features from complex vibration data. Built for scalability, the pipeline is capable of running on local machines for rapid prototyping with synthetic data, as well as on high-end compute clusters (e.g., NVIDIA GPUs with ≥24GB VRAM) for large-scale training.

## Installation & Setup

We recommend using an isolated Python virtual environment to manage dependencies:

```bash
# Initialize and activate the virtual environment
python -m venv .venv
source .venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

*Note: The data preprocessing stage relies on R. Packages such as `dplyr` and `readr` will be automatically resolved when the R pipeline is executed.*

## Usage Guide

The following commands outline the primary workflow, from generating test data to model evaluation.

```bash
# 1. Generate a mock STEAD dataset to verify the pipeline locally
python mock_data.py

# 2. Execute the R pipeline to clean data and generate labels
Rscript data_prep.R

# 3. Train the GNN model using available GPU acceleration
python train.py

# 4. Evaluate predictions to determine accuracy and F1-Score
python evaluate.py
```

## Data Management

The complete Stanford Earthquake Dataset (STEAD) is required for full-scale training. Due to its massive size (~80GB of HDF5 binaries), it is not included in this repository. 

To run the model on a remote instance or server, download the dataset directly via the Kaggle CLI:
```bash
kaggle datasets download -d mostafa/stead
```
Our PyTorch Geometric `Dataset` implementation is optimized to map cleanly onto the `.csv` metadata while dynamically loading required HDF5 waveform chunks. This design strictly manages VRAM utilization during batch processing.

## Project Structure

- **`data_prep.R`** — Handles initial exploratory data analysis, missing value imputation, and robust standardization of raw seismic metadata.
- **`mock_data.py`** — Generates synthetic waveforms that perfectly mimic the STEAD architecture, enabling immediate testing without downloading the 80GB dataset.
- **`dataset.py`** — Contains the core PyTorch Geometric `Dataset` logic to map 3-channel (E, N, Z) seismogram signals into connected graph nodes.
- **`model.py`** — The primary Multimodal Graph Neural Network architecture, combining 1D-CNN temporal feature extractors with Spatial Graph Convolution layers.
- **`train.py`** — The main training loop handling batch ingestion, loss computation (classification & magnitude regression), and backpropagation.
- **`evaluate.py`** — Performance evaluation script calculating standard metrics, including Magnitude MSE and F1-Scores.
