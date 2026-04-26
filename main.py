import argparse
from epgnn.engine.trainer import train_model
from epgnn.engine.evaluator import evaluate_model
from epgnn.data.mock_data import create_mock_stead_data
import os

def main():
    parser = argparse.ArgumentParser(description="EPGNN - Earthquake Prediction Graph Neural Network")
    parser.add_argument('--mode', type=str, required=True, choices=['mock', 'train', 'evaluate', 'smoke'], 
                        help="Mode to run: mock (generate data), train, evaluate, or smoke (run all three)")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    
    args = parser.parse_args()
    
    if args.mode in ['mock', 'smoke']:
        print("=== Generating Mock Data ===")
        create_mock_stead_data()
        
    if args.mode in ['train', 'smoke']:
        print("=== Starting Training ===")
        # Ensure R script ran if needed
        if not os.path.exists('metadata_clean.csv'):
            print("Warning: metadata_clean.csv not found. Please run data_prep.R first.")
        else:
            train_model(epochs=args.epochs)
            
    if args.mode in ['evaluate', 'smoke']:
        print("=== Starting Evaluation ===")
        evaluate_model()

if __name__ == '__main__':
    main()
