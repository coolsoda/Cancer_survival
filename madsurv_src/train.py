#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import random
import json


from .config import *
from .dataset import MADSurvDataset
from .model import MADSurv
from .engine import train_step, eval_step

def run_experiment(args):

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    processed_data_path = Path(PROCESSED_DATA_DIR) / f"{args.dataset_name}_processed_with_folds.csv"
    results_dir = Path(RESULTS_DIR) / args.dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Experiment for: {args.dataset_name} ---")
    print(f"Device: {DEVICE}")
    print(f"Processed data path: {processed_data_path}")
    
    if not processed_data_path.exists():
        print(f"ERROR: Processed data not found at {processed_data_path}")
        print("Please run preprocess.py first for this dataset.")
        return

    fold_metrics = []

    for fold in range(N_SPLITS):
        print(f"\n===== Fold {fold+1}/{N_SPLITS} =====")
        
        train_dataset = MADSurvDataset(processed_data_path, fold=fold, is_train=True)
        val_dataset = MADSurvDataset(processed_data_path, fold=fold, is_train=False)
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )
        
        clinical_dim = len(CLINICAL_FEATURES)
        pathology_dim = len(PATHOLOGY_FEATURES)
        genomic_dim = len(GENOMIC_FEATURES)
        
        model = MADSurv(
            clinical_input_dim=13,
            pathology_input_dim=pathology_dim,
            genomic_input_dim=genomic_dim
        ).to(DEVICE)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        best_c_index = 0.0
        patience_counter = 0
        patience = 5

        for epoch in range(EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
            train_loss = train_step(model, train_dataloader, optimizer, DEVICE)
            val_loss, metrics = eval_step(model, val_dataloader, DEVICE)
            
            scheduler.step() 
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val C-Index: {metrics['c_index']:.4f} | Val Brier Score: {metrics['brier_score']:.4f}")

            if metrics['c_index'] > best_c_index:
                best_c_index = metrics['c_index']
                torch.save(model.state_dict(), results_dir / f"best_model_fold_{fold}.pth")
                print(f"New best C-Index. Model saved for fold {fold}.")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}.")
                    break
        
        model.load_state_dict(torch.load(results_dir / f"best_model_fold_{fold}.pth"))
        _, final_metrics = eval_step(model, val_dataloader, DEVICE)
        fold_metrics.append(final_metrics)
        print(f"--- Final Metrics for Fold {fold+1} ---")
        print(f"C-Index: {final_metrics['c_index']:.4f} | Brier Score: {final_metrics['brier_score']:.4f}")

    metrics_df = pd.DataFrame(fold_metrics)
    mean_c_index = metrics_df['c_index'].mean()
    std_c_index = metrics_df['c_index'].std()
    mean_brier = metrics_df['brier_score'].mean()
    std_brier = metrics_df['brier_score'].std()

    print("\n\n===== Experiment Complete =====")
    print(f"Dataset: {args.dataset_name}")
    print(f"Average C-Index: {mean_c_index:.4f} ± {std_c_index:.4f}")
    print(f"Average Brier Score: {mean_brier:.4f} ± {std_brier:.4f}")

    summary = {
        "dataset": args.dataset_name,
        "mean_c_index": mean_c_index,
        "std_c_index": std_c_index,
        "mean_brier_score": mean_brier,
        "std_brier_score": std_brier,
        "fold_results": metrics_df.to_dict('records')
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Results saved to {results_dir / 'summary.json'}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate the MADSurv model.")
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        required=True,
        choices=['BLCA', 'BRCA', 'HNSC', 'LUAD', 'UCEC'],
        help="Name of the dataset to process (e.g., 'BLCA')."
    )
    args = parser.parse_args()
    
    run_experiment(args)

