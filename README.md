# EPGNN (Earthquake Prediction Graph Neural Network)

EPGNN is a PyTorch research codebase for early-warning event detection and representation learning on multivariate seismic waveforms.

This repository is initialized to run locally for synthetic smoke tests and on a target high-end NVIDIA GPU (≥24GB VRAM) for large-scale training on the STEAD dataset. Datasets are not committed; by default scripts expect the Kaggle STEAD CSV metadata and HDF5 waveform binaries in the project root.

## Quick start

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```
*(Note: R dependencies are automatically installed during the data pipeline step.)*

## Core commands

```bash
# Synthetic end-to-end smoke test
bash scripts/run_smoke_test.sh

# Single-process local GPU training run
python main.py --mode train --epochs 10

# Downstream evaluation skeleton
python main.py --mode evaluate

# Server pretraining skeleton (for real STEAD data)
bash scripts/run_full_train.sh
```

## Server handoff & Datasets

For researchers running the repo on a remote GPU server without project context:

Dataset downloads should be handled manually via the Kaggle API. The Stanford Earthquake Dataset (STEAD) is exceptionally large (~80GB of HDF5 waveforms). 

```bash
kaggle datasets download -d mostafa/stead
```
Our PyTorch Geometric `Dataset` connects directly to the `.csv` metadata files and selectively reads node features from the underlying HDF5 binaries batch-by-batch to strictly manage VRAM utilization.

## Repository layout

* `epgnn/data/` — R-based data cleaning pipelines, PyTorch Geometric Dataset classes, and synthetic mock generators.
* `epgnn/models/` — End-to-end Multimodal Graph Neural Network backbone, containing the 1D-CNN temporal feature extractors and Spatial GCN layers.
* `epgnn/engine/` — Minimal trainers, loss computation (classification & magnitude regression), and evaluators.
* `scripts/` — Helper shell scripts for easy environment setup and execution.
* `main.py` — Centralized CLI entry point for the entire repository.
* `requirements.txt` — PyTorch and data processing dependencies.
