import torch
import torch.nn.functional as F

def reshape_qkv(query_states, key_states, block_size=16): # (bsz, length, num_heads, head_dim) -> (bsz, num_heads, length // chunk_size, chunk_size, head_dim)
    B, H, L, D = query_states.shape
    num_chunks = L // block_size
    
    query_reshaped = query_states.transpose(1, 2)
    key_reshaped = key_states.transpose(1, 2)
    query_reshaped = query_reshaped.reshape(B, H, num_chunks, block_size, D)
    key_reshaped = key_reshaped.reshape(B, H, num_chunks, block_size, D)
    query_reshaped = query_reshaped.reshape(-1, num_chunks, block_size, D)
    key_reshaped = key_reshaped.reshape(-1, num_chunks, block_size, D)

    return query_reshaped, key_reshaped

def baseline_pooling(query_states, key_states, block_size): # ->(bsz, num_heads, L, num_chunks)
    B, H, L, D = query_states.shape
    num_chunks = L // block_size
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    query_states = query_states.reshape(-1, L, D)
    key_states = key_states.reshape(-1, L, D)
    # print("Query and key states reshaped to: ", query_states.shape, key_states.shape)
    attn_weights = torch.einsum("bnd,bmd->bnm", query_states, key_states)
    # print("Attention weights: ", attn_weights.shape)
    attn_weights = apply_causal_mask(attn_weights, 1)
    attn_weights = attn_weights.log_softmax(dim=-1) # for numerical stability
    # print("Attention weights shape after softmax: ", attn_weights.shape)
    attn_weights = attn_weights.reshape(B, H, L, num_chunks, block_size)
    # print("Attention weights shape after reshape: ", attn_weights.shape)
    attn_weights = attn_weights.sum(dim=-1) 
    # print("Attention weights after the sum: ", attn_weights.shape)
    attn_weights = attn_weights.reshape(B, H, L, num_chunks)
    return attn_weights

def compute_divergence(baseline, block_probs, block_size):
    B, Tblocks, D = block_probs.shape
    B, T, D = baseline.shape
    block_probs_reshaped = block_probs.unsqueeze(2).repeat(1, 1, block_size, 1).view(B, Tblocks * block_size, D)
    kl_div = F.kl_div(F.log(baseline + 1e-8), block_probs_reshaped, reduction='mean')
    return kl_div.item()

def expand_blocks(attn_weights, block_size):
    B, num_chunks, _ = attn_weights.shape
    attn_weights = attn_weights.unsqueeze(-2) # B, num_chunks, 1, num_chunks
    attn_weights = attn_weights.repeat(1, 1, block_size, 1) # B, num_chunks, block_size, num_chunks
    attn_weights = attn_weights.reshape(B, num_chunks * block_size, num_chunks)
    return attn_weights

def apply_causal_mask(attn_weights, block_size=16):
    B, L, _ = attn_weights.shape
    mask = torch.triu(torch.ones(L // block_size, L // block_size), diagonal=1).to(attn_weights.device)
    mask = mask.unsqueeze(0)
    mask = mask.repeat(B, 1, 1)
    mask = mask.to(attn_weights.device)
    return attn_weights.masked_fill(mask == 1, float('-inf'))

def transpose_qkv_back(attention_map, block_size=16):
    B, H, L, D = attention_map.shape
    attention_map = attention_map.transpose(1, 2)
    attention_map = attention_map.reshape(B, L, H * D) 
    return attention_map

def compare_divergence(top_blocks_regular, top_blocks_generated, block_size):
    B, H, num_chunks, D = top_blocks_generated.shape
    B, H, T, D = top_blocks_regular.shape
    print("Top blocks generated shape: ", top_blocks_generated.shape)
    print("Top blocks regular: ", top_blocks_regular.shape)
    g_reshaped = top_blocks_generated.unsqueeze(3)
    g_reshaped = g_reshaped.repeat_interleave(block_size, dim=2)
    g_reshaped = g_reshaped.reshape(B, H, T, num_chunks)
    print("G reshaped shape: ", g_reshaped.shape)
    kl_div = F.kl_div(top_blocks_regular, g_reshaped, reduction='mean')
    return kl_div.item()

def avg_pooling(query_states, key_states, block_size):
    B, H, T, D = query_states.shape
    num_chunks = T // block_size
    query_states, key_states = reshape_qkv(query_states, key_states, block_size)
    query_states, key_states = query_states.mean(dim=-2), key_states.mean(dim=-2)
    attn_weights = torch.einsum("bnd,bmd->bnm", query_states, query_states)
    attn_weights = apply_causal_mask(attn_weights, block_size)
    attn_weights = attn_weights.softmax(dim=-1)
    attn_weights = attn_weights.reshape(B, H, num_chunks, num_chunks)
    return attn_weights

def softmax_avg_pooling(query_states, key_states, block_size):
    B, H, T, D = query_states.shape
    num_chunks = T // block_size
    query_states, key_states = reshape_qkv(query_states, key_states, block_size)
    query_states, key_states = query_states.softmax(dim=-1), key_states.softmax(dim=-1)
    query_states, key_states = query_states.mean(dim=-2), key_states.mean(dim=-2)
    attn_weights = torch.einsum("bnd,bmd->bnm", query_states, query_states)
    attn_weights = apply_causal_mask(attn_weights, block_size)
    attn_weights = attn_weights.softmax(dim=-1)
    attn_weights = attn_weights.reshape(B, H, num_chunks, num_chunks)
    return attn_weights

def max_pooling(query_states, key_states, block_size):
    B, H, T, D = query_states.shape
    num_chunks = T // block_size
    query_states, key_states = reshape_qkv(query_states, key_states, block_size)
    query_states, key_states = query_states.max(dim=-2), key_states.max(dim=-2)
    attn_weights = torch.einsum("bnd,bmd->bnm", query_states, query_states)
    attn_weights = apply_causal_mask(attn_weights, block_size)
    attn_weights = attn_weights.softmax(dim=-1)
    attn_weights = attn_weights.reshape(B, H, num_chunks, num_chunks)
    return attn_weights

def softmax_max_pooling(query_states, key_states, block_size):
    B, H, T, _ = query_states.shape
    num_chunks = T // block_size
    query_states, key_states = reshape_qkv(query_states, key_states, block_size)
    query_states, key_states = query_states.softmax(dim=-1), key_states.softmax(dim=-1)
    query_states, _ = query_states.max(dim=-2)
    key_states, _ = key_states.max(dim=-2)
    attn_weights = torch.einsum("bnd,bmd->bnm", query_states, query_states)
    attn_weights = apply_causal_mask(attn_weights, block_size)
    attn_weights = attn_weights.softmax(dim=-1)
    attn_weights = attn_weights.reshape(B, H, num_chunks, num_chunks)
    return attn_weights

def max_softmax_pooling(query_states, key_states, block_size):
    B, H, T, _ = query_states.shape
    num_chunks = T // block_size
    query_states, key_states = reshape_qkv(query_states, key_states, block_size)
    query_states, _ = query_states.max(dim=-2)
    key_states, _ = key_states.max(dim=-2)
    print("Query and key states shape: ", query_states.shape, key_states.shape)
    query_states, key_states = query_states.softmax(dim=-1), key_states.softmax(dim=-1)
    attn_weights = torch.einsum("bnd,bmd->bnm", query_states, query_states)
    attn_weights = apply_causal_mask(attn_weights, block_size)
    attn_weights = attn_weights.softmax(dim=-1)
    attn_weights = attn_weights.reshape(B, H, num_chunks, num_chunks)
    return attn_weights

pooling_methods = {
    "avg_pooling": avg_pooling,
    "max_softmax_pooling": max_softmax_pooling,
    "softmax_max_pooling": softmax_max_pooling,
    "softmax_avg_pooling": softmax_avg_pooling,
    "max_pooling": max_pooling
}