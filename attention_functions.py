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


def softmax_attention(q: torch.Tensor, k: torch.Tensor, v: Optional[torch.Tensor] = None, 
                      causal: bool = True, fp32_attention: bool = True,
                      ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Standard softmax attention; only compute outputs if v is not None
    -> Assume q, k, v are shape (batch_size, num_heads, seq_len, head_dim)
    """
    y = None
    a = torch.einsum('bhmd,bhnd->bhmn', q, k) * (k.shape[-1] ** -0.5)
    if causal:  # Apply causal mask
        m, n = a.shape[-2:]
        causal_mask = torch.ones((m, n), device = a.device, dtype = torch.bool).triu(n - m + 1)
        a = a.masked_fill(causal_mask, -torch.finfo(a.dtype).max)
    if fp32_attention:
        a = torch.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
    else:
        a = torch.softmax(a, dim=-1)
    if v is not None:
        y = torch.einsum('bhmn,bhnd->bhmd', a, v)
    return y, a, None


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

