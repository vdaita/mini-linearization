from torch import nn
import torch
import math
from abc import ABC
import copy
from typing import Optional

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class FeatureMap(nn.Module, ABC):
    def __init__(self, nheads: int, head_dim: int):
        super().__init__()
        self.nheads = nheads
        self.head_dim = head_dim

    def forward(self, query_states: torch.Tensor, key_states: torch.Tensor, attention_mask: torch.Tensor):
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
        return torch.softmax(states, dim=-1)
    
class LinearFeatureMap(FeatureMap):
    def __init__(self, nheads: int, head_dim: int):
        super().__init__(nheads, head_dim)
        self.feature_map = nn.Parameter(torch.randn(nheads, head_dim, head_dim))

    def forward(self, states):
        states = states.transpose(1, 2)
        states = torch.einsum('blhd,hde->blhe', states, self.feature_map)
        states = states.transpose(1, 2)
        return states
    
class LinearAttentionWeights(nn.Module):
    def __init__(self, nheads: int, head_dim: int, mapping_type: str = "linear"):
        super().__init__()
        self.nheads = nheads
        self.head_dim = head_dim
        self.mapping_type = mapping_type

        if mapping_type == "linear":
            self.feature_map_q = LinearFeatureMap(nheads, head_dim)
            self.feature_map_k = LinearFeatureMap(nheads, head_dim)
        elif mapping_type == "hedgehog":
            self.feature_map_q = HedgehogFeatureMap(nheads, head_dim)
            self.feature_map_k = HedgehogFeatureMap(nheads, head_dim)
        else:
            raise ValueError("mapping_type must be either 'linear' or 'hedgehog'")

    def forward(self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        query_states = self.feature_map_k(query_states)
        key_states = self.feature_map_q(key_states)

        attn_weights = torch.einsum('bhld,bhkd->bhlk', query_states, key_states)
        if attention_mask:
            attn_weights = attn_weights.masked_fill(attention_mask < 0, 0)
        else:
            attn_weights = attn_weights.tril(diagonal=1)

        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
        return attn_weights
    
    def get_name(self):
        return f"linear_map_{self.mapping_type}"
    


class DiffLinearAttentionWeights(nn.Module):
    def __init__(self, n_heads: int, head_dim: int, depth: int, map_type_1: str = "linear", map_type_2: str = None):
        super().__init__()
        self.depth = depth
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.map_type_1 = map_type_1
        self.map_type_2 = map_type_2 

        if not map_type_2:
            map_type_2 = map_type_1

        self.la_1 = LinearAttentionWeights(n_heads, head_dim, map_type_1)
        self.la_2 = LinearAttentionWeights(n_heads, head_dim, map_type_2)

        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim) * 0.1)

        self.lambda_init = lambda_init_fn(self.depth)

    def forward(self, 
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: torch.Tensor
    ):
        B, H, T, D = query_states.shape
        assert H == self.n_heads and D == self.head_dim

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(query_states)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(query_states)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attn_weights_1 = self.la_1(query_states, key_states, attention_mask)
        attn_weights_2 = self.la_2(query_states, key_states, attention_mask)

        return attn_weights_1 - lambda_full * attn_weights_2
    
    def get_name(self):
        return f"diff_linear_map_{self.map_type_1}_{self.map_type_2}"
    
if __name__ == "__main__":
    print("Testing attention methods...")
    batch_size = 4
    n_heads = 4
    head_dim = 64
    depth = 2
    seq_len = 10

    la_regular = LinearAttentionWeights(n_heads, head_dim, "linear")
    print(la_regular(
        torch.randn(batch_size, n_heads, seq_len, head_dim),
        torch.randn(batch_size, n_heads, seq_len, head_dim),
        None
    ))
    print("Regular: ", la_regular.get_name())
    la_diff = DiffLinearAttentionWeights(n_heads, head_dim, depth, "linear", "hedgehog")
    print(la_diff(
        torch.randn(batch_size, n_heads, seq_len, head_dim),
        torch.randn(batch_size, n_heads, seq_len, head_dim),
        None
    ))
    print(la_diff.get_name())
