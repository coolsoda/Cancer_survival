#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
from lifelines.utils import concordance_index
from sklearn.metrics import brier_score_loss

def nll_loss(hazards: torch.Tensor, survival: torch.Tensor, target: torch.Tensor):
    """
    Custom Negative Log-Likelihood loss function for discrete-time survival analysis.
    """
    # Get the interval index (j_i) and event indicator (c_i) for each patient in the batch
    j_i = target[:, 0]
    c_i = target[:, 1]
    
    s_j_minus_1 = torch.cat([
        torch.ones(survival.shape[0], 1, device=survival.device), 
        survival[:, :-1]
    ], dim=1)
    s_j_minus_1_selected = torch.gather(s_j_minus_1, 1, j_i.unsqueeze(1)).squeeze()

    h_j_selected = torch.gather(hazards, 1, j_i.unsqueeze(1)).squeeze()
    
    s_j_selected = torch.gather(survival, 1, j_i.unsqueeze(1)).squeeze()

    likelihood = torch.where(
        c_i == 1,
        s_j_minus_1_selected * h_j_selected,
        s_j_selected
    )
    

    likelihood = torch.clamp(likelihood, min=1e-15)
    
    loss = -torch.log(likelihood).mean()
    return loss

def get_brier_score(hazards: np.ndarray, target: np.ndarray) -> float:
    """
    Brier Score.
    
    """
    survival_probs = np.cumprod(1 - hazards, axis=1)
    
    j_i = target[:, 0] 
    c_i = target[:, 1] 
    
    n_samples, n_intervals = survival_probs.shape
    ibs = 0.0
    
    for j in range(n_intervals):
        # True status at time j+1
        # A patient survived past interval j if their event/censor time is in a later interval that does not occur yet
        y_true = (j_i > j).astype(int) 
        y_pred = survival_probs[:, j]
        
        # We can only calculate Brier score for patients who are still at risk at time j
        
        mask = (j_i >= j)
        if mask.sum() > 0:
            brier = brier_score_loss(y_true[mask], y_pred[mask])
            ibs += brier
            
    return ibs / n_intervals

def get_concordance_index(hazards: np.ndarray, target: np.ndarray) -> float:
    """
    Concordance Index (C-index).

    """
    
    risk_scores = hazards.sum(axis=1)
    
    durations = target[:, 0]
    events = target[:, 1]
    
    return concordance_index(durations, -risk_scores, events)

