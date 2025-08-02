#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from .modules import MLPEncoder, TransformerEncoder, UncertaintyAwareAttentionGate, PredictionHead
from .config import (
    EMBEDDING_DIM, K_INTERVALS, T_DROPOUT,
    TRANSFORMER_LAYERS, TRANSFORMER_HEADS, TRANSFORMER_HIDDEN_DIM
)

class MADSurv(nn.Module):

    def __init__(self, clinical_input_dim: int, pathology_input_dim: int, genomic_input_dim: int):
        super().__init__()
        
        self.clinical_encoder = MLPEncoder(input_dim=clinical_input_dim, output_dim=EMBEDDING_DIM)
        self.pathology_encoder = MLPEncoder(input_dim=pathology_input_dim, output_dim=EMBEDDING_DIM)
        self.genomic_encoder = TransformerEncoder(
            input_dim=genomic_input_dim,
            output_dim=EMBEDDING_DIM,
            n_layers=TRANSFORMER_LAYERS,
            n_heads=TRANSFORMER_HEADS,
            hidden_dim=TRANSFORMER_HIDDEN_DIM
        )
        
        self.attention_gate = UncertaintyAwareAttentionGate(embedding_dim=EMBEDDING_DIM)
        
        # Prediction head
        self.prediction_head = PredictionHead(input_dim=EMBEDDING_DIM, k_intervals=K_INTERVALS)

    def _get_stochastic_embeddings(self, encoder: nn.Module, x: torch.Tensor, T: int) -> (torch.Tensor, torch.Tensor):
        """
        Performs T forward passes with dropout enabled to get mean embeddings and uncertainty.
        """
        encoder.train() 
        
        embeddings = []
        for _ in range(T):
            with torch.no_grad():
                emb = encoder(x)
                embeddings.append(emb)
        
        embeddings = torch.stack(embeddings, dim=0)
        
        mean_embedding = embeddings.mean(dim=0)
        uncertainty = embeddings.var(dim=0).mean(dim=1, keepdim=True) # Scalar uncertainty per sample
        
        return mean_embedding, uncertainty

    def forward(self, clinical_x, genomic_x, pathology_x, T_dropout=T_DROPOUT, is_train=True):
        """
        Defines the forward pass of the MADSurv model.
        """
        if is_train:
            z_c = self.clinical_encoder(clinical_x)
            z_g = self.genomic_encoder(genomic_x)
            z_p = self.pathology_encoder(pathology_x)
            
        z_c, u_c = self._get_stochastic_embeddings(self.clinical_encoder, clinical_x, T_dropout)
        z_g, u_g = self._get_stochastic_embeddings(self.genomic_encoder, genomic_x, T_dropout)
        z_p, u_p = self._get_stochastic_embeddings(self.pathology_encoder, pathology_x, T_dropout)
        
        attention_weights = self.attention_gate(z_c, z_g, z_p, u_c, u_g, u_p)
        
        alpha_c = attention_weights[:, 0].unsqueeze(1)
        alpha_g = attention_weights[:, 1].unsqueeze(1)
        alpha_p = attention_weights[:, 2].unsqueeze(1)
        
        z_fused = (alpha_c * z_c) + (alpha_g * z_g) + (alpha_p * z_p)
        
        hazard_probs = self.prediction_head(z_fused)
        
        return hazard_probs, attention_weights

