#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RAW_DATA_DIR = "../data/raw/"
PROCESSED_DATA_DIR = "../data/processed/"
NUM_WORKERS = 4 
SEED = 42

# To match the column names in data files
CLINICAL_FEATURES = [
    'age', 'weight', 'race', 'tumor_grade', 'treatment', 
    'aneuploidy_score', 'msi_score', 'tmb', 
    'tumor_necrosis_pct', 'tumor_nuclei_pct'
]
GENOMIC_FEATURES = [f"pathway_{i}" for i in range(331)] # Example names
PATHOLOGY_FEATURES = [
    'lymphocytes', 'monocytes', 'necrotic_tissue', 'neutrophils', 
    'non_cancerous_cells', 'cancerous_cells', 'tumor_cell_nuclei_area', 
    'supportive_tissue'
]
TARGET_DURATION_COL = 'duration'
TARGET_EVENT_COL = 'event'


# Model & Training Hyperparameters
N_SPLITS = 5                # 5-fold
EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
EMBEDDING_DIM = 256 
K_INTERVALS = 15            
T_DROPOUT = 30              

# Encoder settings for genomic data
TRANSFORMER_LAYERS = 4
TRANSFORMER_HEADS = 4
TRANSFORMER_HIDDEN_DIM = 256

