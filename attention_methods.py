from torch import nn
import torch
import math
from abc import ABC
import copy
from typing import Optional
from attention_functions import linear_attention, quadratic_attention, diff_linear_attention, diff_quadratic_attention

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
        self.feature_map = nn.Parameter(torch.rand(nheads, head_dim, head_dim // 2))

    def forward(self, states): # states: (B, H, T, D)
        states = states.transpose(1, 2)
        # print("Hedgehog feature map shapes: ", states.shape, self.feature_map.shape)
        states = torch.einsum('blhd,hde->blhe', states, self.feature_map)
        states = torch.cat([states, -states], dim=-1)
        states = states.transpose(1, 2)
        states = torch.softmax(states, dim=-1)
        return states
    
class LinearFeatureMap(FeatureMap):
    def __init__(self, nheads: int, head_dim: int):
        super().__init__(nheads, head_dim)
        self.feature_map = nn.Parameter(torch.rand(nheads, head_dim, head_dim))

    def forward(self, states):
        states = states.transpose(1, 2)
        # print("Linear feature map shapes: ", states.shape, self.feature_map.shape)
        states = torch.einsum('blhd,hde->blhe', states, self.feature_map)
        states = states.transpose(1, 2)
        states = states.relu()
        return states
    
class LinearAttention(nn.Module):
    def __init__(self, nheads: int, head_dim: int, fm: str = 'linear'):
        super().__init__()
        self.nheads, self.head_dim, self.fm = nheads, head_dim, fm

        if fm == 'linear':
            self.fm_query, self.fm_key = LinearFeatureMap(nheads, head_dim), LinearFeatureMap(nheads, head_dim)
        else:
            self.fm_query, self.fm_key = HedgehogFeatureMap(nheads, head_dim), HedgehogFeatureMap(nheads, head_dim)

    def forward(self,
                query_states,
                key_states,
                value_states,
                implementation: str = 'quadratic'):
        query_states, key_states = self.fm_query(query_states), self.fm_key(key_states)
        if implementation == 'linear':
            return linear_attention(query_states, key_states, value_states)
        else:
            return quadratic_attention(query_states, key_states, value_states)
        
    def get_name(self):
        return f"LinearAttention_{self.fm}"

class DiffLinearAttention(nn.Module):
    def __init__(self, nheads: int, head_dim: int, fm_1: str = 'linear', fm_2: str = 'linear'):
        super().__init__()

        self.fm_1, self.fm_2, self.nheads, self.head_dim = fm_1, fm_2, nheads, head_dim

        if fm_1 == 'linear':
            self.fm_1_query, self.fm_1_key = LinearFeatureMap(nheads, head_dim), LinearFeatureMap(nheads, head_dim)
        else:
            self.fm_1_query, self.fm_1_key = HedgehogFeatureMap(nheads, head_dim), HedgehogFeatureMap(nheads, head_dim)

        if fm_2 == 'linear':
            self.fm_2_query, self.fm_2_key = LinearFeatureMap(nheads, head_dim), LinearFeatureMap(nheads, head_dim)
        else:
            self.fm_2_query, self.fm_2_key = HedgehogFeatureMap(nheads, head_dim), HedgehogFeatureMap(nheads, head_dim)
        
        self.alpha = nn.Parameter(torch.Tensor([0.3]))

        
    def forward(self,
                query_states,
                key_states,
                value_states,
                implementation: str = 'quadratic'):
        query_states_1, key_states_1 = self.fm_1_query(query_states), self.fm_1_key(key_states)
        query_states_2, key_states_2 = self.fm_2_query(query_states).sigmoid(), self.fm_2_key(key_states).sigmoid()

        query_states_2 *= query_states_1
        key_states_2 *= key_states_1

        if implementation == 'linear':
            return diff_linear_attention(query_states_1, key_states_1, query_states_2, key_states_2, value_states, self.alpha)
        else:
            # print("Using quadratic attention")
            return diff_quadratic_attention(query_states_1, key_states_1, query_states_2, key_states_2, value_states, self.alpha)
        
    def get_name(self):
        return f"DiffLinearAttention_{self.fm_1}_{self.fm_2}"

if __name__ == "__main__":
    print("Testing attention methods...")
    batch_size = 1
    n_heads = 1
    head_dim = 4
    depth = 2
    seq_len = 4

    query_states_1 = torch.rand(batch_size, n_heads, seq_len, head_dim).relu()
    key_states_1 = torch.rand(batch_size, n_heads, seq_len, head_dim).relu()

    query_states_2 = torch.rand(batch_size, n_heads, seq_len, head_dim).relu().sigmoid()
    key_states_2 = torch.rand(batch_size, n_heads, seq_len, head_dim).relu().sigmoid()

    query_states_2 *= query_states_1
    key_states_2 *= query_states_2

    value_states = torch.rand(batch_size, n_heads, seq_len, head_dim).relu()
    alpha = torch.Tensor([0.8])

    linear_attention_result, _, _ = linear_attention(query_states_1, key_states_1, value_states)
    quadratic_attention_result, attention_weights, _ = quadratic_attention(query_states_1, key_states_1, value_states)
    print(torch.allclose(linear_attention_result, quadratic_attention_result))
    print(attention_weights)
    print(attention_weights.sum(dim=-1))

    linear_attention_result, _, _ = diff_linear_attention(query_states_1, key_states_1, query_states_2, key_states_2, value_states, alpha)
    quadratic_attention_result, attention_weights, _ = diff_quadratic_attention(query_states_1, key_states_1, query_states_2, key_states_2, value_states, alpha)
    print(torch.allclose(linear_attention_result, quadratic_attention_result))
    print(attention_weights.sum(dim=-1))

    print("=============")

    dla_module = DiffLinearAttention(n_heads, head_dim, 'linear', 'linear')
    dla_result, dla_attention, _ = dla_module(query_states_1, key_states_1, value_states)

    la_module = LinearAttention(n_heads, head_dim, 'linear')
    la_result, la_attention, _ = la_module(query_states_1, key_states_1, value_states)

    print("Diff linear attention: ", dla_result)
    print("Diff linear attention map: ", dla_attention)
    print("Linear attention result: ", la_result)
    print("Linear attention map: ", la_attention)