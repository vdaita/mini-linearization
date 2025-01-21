# From: https://github.com/HazyResearch/lolcats/blob/main/src/model/linear_attention/linear_attention.py
from typing import Optional, Tuple
import torch
try:
    from fast_transformers.causal_product import causal_dot_product as fast_causal_dot_product
except ImportError:
    fast_causal_dot_product = None

def causal_dot_product(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    Causal linear attention dot product
    - If available, use CUDA kernel from fast-transformers
    """
    if fast_causal_dot_product is None:
        kv = torch.einsum('bhlf,bhld->bhlfd', k, v)
        return torch.einsum('bhlf,bhlfd->bhld', q, kv.cumsum(dim=2))
    return fast_causal_dot_product(q, k, v)

def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     fp32_attention: bool = False, eps: float = 1e-12,
                     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Compute linear attention with CUDA kernel implementation from fast-transformers
    - https://github.com/idiap/fast-transformers
    - Assume q, k are shape (batch_size, num_heads, seq_len, feature_dim); 
      v is shape (b, h, l, head_dim)
    """
    dtype = q.dtype
    # Causal mask already applied
    y = causal_dot_product(q.contiguous().to(dtype=torch.float32),
                           k.contiguous().to(dtype=torch.float32),
                           v.contiguous().to(dtype=torch.float32))
    if fp32_attention:
        y = (y / (torch.einsum(
            "bhld,bhld->bhl", q.float(), k.float().cumsum(dim=2)
        ) + eps)[..., None]).to(dtype=dtype)
    else:
        y = y.to(dtype=dtype)
        k = k.float().cumsum(dim=2).to(dtype=dtype)
        y = y / (torch.einsum("bhld,bhld->bhl", q, k) + eps)[..., None]
    return y, None, None

def diff_linear_attention(q_1: torch.Tensor, k_1: torch.Tensor, q_2: torch.Tensor, k_2: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor,
                     fp32_attention: bool = False, eps: float = 1e-12,
                     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Compute linear attention with CUDA kernel implementation from fast-transformers
    - https://github.com/idiap/fast-transformers
    - Assume q, k are shape (batch_size, num_heads, seq_len, feature_dim); 
      v is shape (b, h, l, head_dim)
    """
    dtype = q_1.dtype

    if not(0 <= alpha.item() and alpha.item() <= 1):
        breakpoint()
    # Causal mask already applied
    y_1 = causal_dot_product(q_1.contiguous().to(dtype=torch.float32),
                           k_1.contiguous().to(dtype=torch.float32),
                           v.contiguous().to(dtype=torch.float32))
      
    y_2 = causal_dot_product(q_2.contiguous().to(dtype=torch.float32),
                           k_2.contiguous().to(dtype=torch.float32),
                           v.contiguous().to(dtype=torch.float32))

    sum_1 = (torch.einsum(
        "bhld,bhld->bhl", q_1.float(), k_1.float().cumsum(dim=2)
    ) + eps)[..., None]
    sum_2 = (torch.einsum(
        "bhld,bhld->bhl", q_2.float(), k_2.float().cumsum(dim=2)
    ) + eps)[..., None]

    y = (y_1 - alpha * y_2) / (sum_1 - alpha * sum_2)

    return y, None, None

def quadratic_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor = None,
                        causal: bool = True, fp32_attention: bool = False, eps: float = 1e-12,
                        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Compute attention with feature maps by instantiating L x L matrix of attention weights
    -> Use for attention distillation
    -> Assume q, k are shape (batch_size, num_heads, seq_len, feature_dim); v is shape (b, h, l, head_dim)
    """
    y = None
    dtype = q.dtype
    if fp32_attention:
        q, k = q.float(), k.float()
    a = torch.einsum('bhmd,bhnd->bhmn', q, k)  # note we don't scale, tho we could
    if causal:  # Apply causal mask
        m, n = a.shape[-2:]
        causal_mask = torch.ones((m, n), device = a.device, dtype = torch.bool).triu(n - m + 1)
        a = a.masked_fill(causal_mask, 0)
    # Normalize to compute attention
    a = a / (a.sum(dim=-1, keepdim=True) + eps)
    a = a.to(dtype=dtype) if fp32_attention else a
    if torch.isnan(a).sum() > 0:
        breakpoint()
    if v is not None:
        y = torch.einsum('bhmn,bhnd->bhmd', a, v)
    return y, a, None

def diff_quadratic_attention(q_1: torch.Tensor, k_1: torch.Tensor, q_2: torch.Tensor, k_2: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor,
                        causal: bool = True, fp32_attention: bool = False, eps: float = 1e-12,
                        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Compute attention with feature maps by instantiating L x L matrix of attention weights
    -> Use for attention distillation
    -> Assume q, k are shape (batch_size, num_heads, seq_len, feature_dim); v is shape (b, h, l, head_dim)
    """
    y = None
    dtype = q_1.dtype
    if fp32_attention:
        q_1, k_1 = q_1.float(), k_1.float()
        q_2, k_2 = q_2.float(), k_2.float()
    
    if not(0 <= alpha.item() and alpha.item() <= 1):
        breakpoint()

    a_1 = torch.einsum('bhmd,bhnd->bhmn', q_1, k_1)  # note we don't scale, tho we could
    a_2 = torch.einsum('bhmd,bhnd->bhmn', q_2, k_2)

    if causal:  # Apply causal mask
        m, n = a_1.shape[-2:]
        causal_mask = torch.ones((m, n), device = a_1.device, dtype = torch.bool).triu(n - m + 1)
        a_1 = a_1.masked_fill(causal_mask, 0)
        a_2 = a_2.masked_fill(causal_mask, 0)

    a_1 = a_1
    a_1 = a_1.to(dtype=dtype) if fp32_attention else a_1

    a_2 = a_2
    a_2 = a_2.to(dtype=dtype) if fp32_attention else a_2

    # Normalize to compute attention
    if torch.isnan(a_1).sum() > 0 or torch.isnan(a_2).sum() > 0:
        breakpoint()

    a = a_1 - alpha * a_2
    a /= a_1.sum(dim=-1, keepdim=True) - alpha * a_2.sum(dim=-1, keepdim=True) + eps
    if v is not None:
        y = torch.einsum('bhmn,bhnd->bhmd', a, v)
    return y, a, None