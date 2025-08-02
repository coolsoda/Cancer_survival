#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .model import MADSurv
from .utils import nll_loss, get_brier_score, get_concordance_index

def train_step(
    model: MADSurv,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str
):

    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        clinical = batch['clinical'].to(device)
        genomic = batch['genomic'].to(device)
        pathology = batch['pathology'].to(device)
        target = batch['target'].to(device)

        hazards, _ = model(clinical, genomic, pathology)
        
        survival = torch.cumprod(1 - hazards, dim=1)
        
        # Calculate loss
        loss = nll_loss(hazards, survival, target)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


def eval_step(
    model: MADSurv,
    dataloader: DataLoader,
    device: str
):
    """A single evaluation step."""
    model.eval()
    total_loss = 0.0
    
    all_hazards = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            clinical = batch['clinical'].to(device)
            genomic = batch['genomic'].to(device)
            pathology = batch['pathology'].to(device)
            target = batch['target'].to(device)

            hazards, _ = model(clinical, genomic, pathology)
            
            survival = torch.cumprod(1 - hazards, dim=1)
            
            loss = nll_loss(hazards, survival, target)
            total_loss += loss.item()
            
            all_hazards.append(hazards.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            
    all_hazards = np.concatenate(all_hazards, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    c_index = get_concordance_index(all_hazards, all_targets)
    brier_score = get_brier_score(all_hazards, all_targets)
    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, {"c_index": c_index, "brier_score": brier_score}

