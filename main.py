import argparse
from engine.trainer import train_model
from engine.evaluator import evaluate_model
from data.mock_data import create_mock_stead_data
import os

def main():
    parser = argparse.ArgumentParser(description="EPGNN - Earthquake Prediction Graph Neural Network")
    parser.add_argument('--mode', type=str, required=True, choices=['mock', 'train', 'evaluate', 'smoke'], 
                        help="Mode to run: mock (generate data), train, evaluate, or smoke (run all three)")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--metadata_path', type=str, default='metadata_clean.csv', help="Path to cleaned metadata CSV")
    parser.add_argument('--hdf5_path', type=str, default='mock_waveforms.hdf5', help="Path to HDF5 waveform binary")
    
    args = parser.parse_args()
    
    if args.mode in ['mock', 'smoke']:
        print("=== Generating Mock Data ===")
        create_mock_stead_data()
        
    if args.mode in ['train', 'smoke']:
        print("=== Starting Training ===")
        # Ensure R script ran if needed
        if not os.path.exists(args.metadata_path):
            print(f"Warning: {args.metadata_path} not found. Please run data_prep.R first.")
        else:
            train_model(epochs=args.epochs, metadata_path=args.metadata_path, hdf5_path=args.hdf5_path)
            
    if args.mode in ['evaluate', 'smoke']:
        print("=== Starting Evaluation ===")
        evaluate_model(metadata_path=args.metadata_path, hdf5_path=args.hdf5_path)

if __name__ == '__main__':
    main()
