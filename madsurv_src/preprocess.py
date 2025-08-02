#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import argparse
from pathlib import Path


RAW_CLINICAL_FEATURES = [
    'age', 'weight', 'race', 'tumor_grade', 'treatment', 'aneuploidy_score', 
    'msi_score', 'tmb', 'tumor_necrosis_pct', 'tumor_nuclei_pct'
]
CONTINUOUS_COLS = [
    'age', 'weight', 'aneuploidy_score', 'msi_score', 'tmb', 
    'tumor_necrosis_pct', 'tumor_nuclei_pct'
]
ORDINAL_COLS = ['tumor_grade']
ONEHOT_COLS = ['race']

# The first treatment information after diagnosis is one-hot encoded
BINARY_COLS = ['treatment']

PATHOLOGY_FEATURES = [
    'lymphocytes', 'monocytes', 'necrotic_tissue', 'neutrophils', 
    'non_cancerous_cells', 'cancerous_cells', 'tumor_cell_nuclei_area', 
    'supportive_tissue'
]

def preprocess_data(raw_data_path: Path, output_path: Path, n_splits: int = 5, seed: int = 42):
    """
    Loads raw clinical, genomic, and pathology data, preprocesses, merges, and creates stratified folds.
    """
    print("Starting data preprocessing...")

    # Data is in separate CSV files, each with a 'patient_id' column
    clinical_df = pd.read_csv(raw_data_path / "clinical_raw.csv")
    genomic_df = pd.read_csv(raw_data_path / "genomic_raw.csv")
    pathology_df = pd.read_csv(raw_data_path / "pathology_raw.csv")
    
    patient_id_col = 'patient_id'
    duration_col = 'duration'
    event_col = 'event'
    

    # Standardize continuous features
    scaler = StandardScaler()
    clinical_df[CONTINUOUS_COLS] = scaler.fit_transform(clinical_df[CONTINUOUS_COLS])
    
    # One-hot encoding for categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    onehot_encoded = encoder.fit_transform(clinical_df[ONEHOT_COLS])
    onehot_df = pd.DataFrame(onehot_encoded, columns=encoder.get_feature_names_out(ONEHOT_COLS))
    
    # Combine
    processed_clinical = pd.concat([
        clinical_df[[patient_id_col, duration_col, event_col] + ORDINAL_COLS + BINARY_COLS],
        clinical_df[CONTINUOUS_COLS],
        onehot_df
    ], axis=1)


    min_max_scaler = MinMaxScaler()
    pathology_df[PATHOLOGY_FEATURES] = min_max_scaler.fit_transform(pathology_df[PATHOLOGY_FEATURES])
    processed_pathology = pathology_df


    processed_genomic = genomic_df
    
    print("Merging all data modalities...")
    df_merged = pd.merge(processed_clinical, processed_genomic, on=patient_id_col)
    df_merged = pd.merge(df_merged, processed_pathology, on=patient_id_col)
    
    print(f"Creating {n_splits} stratified folds...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df_merged['fold'] = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_merged, df_merged[event_col])):
        df_merged.loc[val_idx, 'fold'] = fold
        
    # Optional: save preprocessed data
    output_path.mkdir(parents=True, exist_ok=True)
    output_filepath = output_path / "processed_data_with_folds.csv"
    df_merged.to_csv(output_filepath, index=False)
    print(f"Preprocessing complete. Final data saved to {output_filepath}")
    print("\nFinal DataFrame head:")
    print(df_merged.head())
    print(f"\nValue counts for folds:\n{df_merged['fold'].value_counts()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess multimodal cancer survival data.")
    parser.add_argument(
        '--raw_data_path', 
        type=str, 
        default="../dataset_csv/",
        help="Path to the directory containing raw data files."
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default="../dataset_csv/processed/",
        help="Path to the directory to save the processed data."
    )
    args = parser.parse_args()
    
    preprocess_data(Path(args.raw_data_path), Path(args.output_path))

