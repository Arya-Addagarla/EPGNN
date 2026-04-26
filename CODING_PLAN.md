# Coding Plan

## Phase 1: MVP Architecture (Completed)
- Scaffold basic R data cleaning pipelines (`data_prep.R`)
- Implement custom `torch_geometric` Dataset for HDF5 extraction (`dataset.py`)
- Build baseline `MultimodalGNN` with 1D-CNN extractors (`gnn.py`)

## Phase 2: Restructuring (Completed)
- Flatten architecture to match top-tier academic repositories.
- Abstract training/evaluation engines.
- Provide automated shell scripts for dataset fetching and smoke testing.

## Phase 3: Scaling & Hardware Optimization (Pending)
- Refactor `dataset.py` to use `h5py` chunk caching for better VRAM utilization on 24GB+ GPUs.
- Implement Distributed Data Parallel (DDP) for multi-GPU server clusters.
- Integrate optional fiber-optic and GPS secondary modalities.
