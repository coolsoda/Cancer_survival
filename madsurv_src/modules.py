#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    """A MLP encoder for clinical and pathology data."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = [64, 128], dropout: float = 0.3):
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class TransformerEncoder(nn.Module):
    """A Transformer-based encoder for genomic pathway data."""
    def __init__(self, input_dim: int, output_dim: int, n_layers: int, n_heads: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        # Project each scalar pathway score into a token embedding
        self.embedding = nn.Linear(1, hidden_dim) 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Final layer to project the aggregated features to the output dimension
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.unsqueeze(-1) 
        
        tokens = self.embedding(x) 
        
        encoded_tokens = self.transformer(tokens) 
        
        aggregated = encoded_tokens.mean(dim=1)
        
        return self.output_layer(aggregated)

class UncertaintyAwareAttentionGate(nn.Module):
    """
    An attention gate that uses both features and uncertainty
    to produce modality weights.
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        # Input to the gating network: 3 embeddings with corresponding uncertainty scores
        input_size = (embedding_dim * 3) + 3
        self.gate_network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 3) # 3 logits, one for each modality
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z_c, z_g, z_p, u_c, u_g, u_p) -> torch.Tensor:

        evidence_vector = torch.cat([z_c, z_g, z_p, u_c, u_g, u_p], dim=1)
        
        logits = self.gate_network(evidence_vector)
        
        attention_weights = self.softmax(logits)
        
        return attention_weights

class PredictionHead(nn.Module):
    """A final MLP to predict hazard probabilities from the fused vector."""
    def __init__(self, input_dim: int, k_intervals: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, k_intervals),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

