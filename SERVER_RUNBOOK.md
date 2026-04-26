# Server Runbook

Instructions for deploying EPGNN on remote high-performance compute clusters (e.g., AWS EC2, AMD MI300X, or NVIDIA A100 nodes).

## 1. Initial Setup
```bash
git clone https://github.com/YOUR_USERNAME/EPGNN.git
cd EPGNN
bash scripts/setup_env.sh
```

## 2. Dataset Acquisition
If the server does not have the STEAD dataset pre-downloaded, execute:
```bash
bash scripts/download_datasets.sh
```
*Note: Ensure `~/.kaggle/kaggle.json` is configured with valid API credentials on the server.*

## 3. Training Execution
To launch the primary training loop inside a `tmux` or `screen` session:
```bash
nohup bash scripts/run_full_train.sh > training_log.txt 2>&1 &
```
