from torch import nn
import torch
import math
from abc import ABC
import copy
from typing import Optional
from fast_transformers.causal_product import causal_dot_product
from attention_functions import linear_attention, softmax_attention, quadratic_attention

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class FeatureMap(nn.Module, ABC):
    def __init__(self, nheads: int, head_dim: int):
        super().__init__()
        self.nheads = nheads
        self.head_dim = head_dim

    def forward(self, states: torch.Tensor):
        raise NotImplementedError("Subclasses should implement this method")

class HedgehogFeatureMap(FeatureMap):
    def __init__(self, nheads: int, head_dim: int):
        super().__init__(nheads, head_dim)
        self.feature_map = nn.Parameter(torch.randn(nheads, head_dim, head_dim // 2))

    def forward(self, states): # states: (B, H, T, D)
        states = states.transpose(1, 2)
        states = torch.einsum('blhd,hde->blhe', states, self.feature_map)
        states = torch.cat([states, -states], dim=-1)
        states = states.transpose(1, 2)
        states = torch.softmax(states, dim=-1)
        return states
    
class LinearFeatureMap(FeatureMap):
    def __init__(self, nheads: int, head_dim: int):
        super().__init__(nheads, head_dim)
        self.feature_map = nn.Parameter(torch.randn(nheads, head_dim, head_dim))

    def forward(self, states):
        states = states.transpose(1, 2)
        states = torch.einsum('blhd,hde->blhe', states, self.feature_map)
        states = states.transpose(1, 2)
        states = states.relu()
        return states

if __name__ == "__main__":
    print("Testing attention methods...")
    batch_size = 1
    n_heads = 1
    head_dim = 4
    depth = 2
    seq_len = 4

    query_states = torch.randn(batch_size, n_heads, seq_len, head_dim)
    key_states = torch.randn(batch_size, n_heads, seq_len, head_dim)
    value_states = torch.randn(batch_size, n_heads, seq_len, head_dim)

    linear_attention_result, _, _ = linear_attention(query_states, key_states, value_states)
    quadratic_attention_result, _, _ = quadratic_attention(query_states, key_states, value_states)
    print(torch.allclose(linear_attention_result, quadratic_attention_result))