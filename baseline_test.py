import torch
import os
# Get terminal width
terminal_width = os.get_terminal_size().columns
torch.set_printoptions(linewidth=terminal_width, sci_mode=False, precision=4)

torch.manual_seed(42)

B = 1
H = 1
T = 8
D = 8

block_size = 2

test_q = torch.randn(B, H, T, D)
test_k = torch.randn(B, H, T, D)

def apply_causal_mask(attn_weights):
    B, L, _ = attn_weights.shape
    mask = torch.triu(torch.ones(L, L), diagonal=1).to(attn_weights.device)
    mask = mask.unsqueeze(0)
    mask = mask.repeat(B, 1, 1)
    mask = mask.to(attn_weights.device)
    return attn_weights.masked_fill(mask == 1, float('-inf'))

def baseline_attention(q, k):
    print("=====Running Baseline Attention=====")
    B, H, T, D = q.shape
    q, k = q.reshape(B * H, T, D), k.reshape(B * H, T, D)
    print("QK shape: ", q.shape, k.shape)
    attn_weights = torch.einsum('bnd,bmd->bnm', q, k)
    attn_weights = apply_causal_mask(attn_weights)
    attn_weights = torch.softmax(attn_weights, dim=-1)
    print("Attn weights shape: ", attn_weights.shape)
    return attn_weights

def averaged_attention(q, k):
    print("=====Running Averaged Attention=====")
    B, H, T, D = q.shape
    q, k = q.reshape(B * H, T, D), k.reshape(B * H, T, D)
    print("QK shape: ", q.shape, k.shape)
    q, k = q.reshape(B * H, T // block_size, block_size, D), k.reshape(B * H, T // block_size, block_size, D)
    q, k = q.mean(dim=2), k.mean(dim=2)
    attn_weights = torch.einsum('bnd,bmd->bnm', q, k)
    attn_weights = apply_causal_mask(attn_weights)
    attn_weights = torch.softmax(attn_weights, dim=-1)
    print("Attn weights shape: ", attn_weights.shape)
    return attn_weights

def compare_attention(q, k):
    B, H, T, D = q.shape
    num_chunks = T // block_size

    baseline_weights = baseline_attention(q, k)
    averaged_weights = averaged_attention(q, k)

    print("baseline weights: ", baseline_weights)

    baseline_weights = baseline_weights.reshape(B * H, T, num_chunks, block_size)
    baseline_weights = baseline_weights.sum(dim=-1)

    print("baseline weights shape: ", baseline_weights.shape)
    print("baseline weights with mean: ", baseline_weights)
    print("averaged weights: ", averaged_weights)

    num_selected = 1

    top_averaged_blocks = torch.topk(averaged_weights, num_selected, dim=-1).indices
    print("Top averaged blocks before repeat: ", top_averaged_blocks)
    top_averaged_blocks = torch.repeat_interleave(top_averaged_blocks, block_size, dim=-2)
    print("Top averaged blocks: ", top_averaged_blocks)

    blocks_selected = torch.zeros(B * H, T, num_selected)

    # Create a tensor for the result
    blocks_selected = torch.empty(B * H, T, num_selected, device=baseline_weights.device)

    # Use advanced indexing to parallelize
    i = torch.arange(B * H).unsqueeze(1).unsqueeze(2)  # Shape: (B * H, 1, 1)
    j = torch.arange(T).unsqueeze(0).unsqueeze(2)    # Shape: (1, T, 1)

    # Gather selected blocks based on indices
    blocks_selected = baseline_weights[i, j, top_averaged_blocks]

    weight_percentage_met = blocks_selected.sum(dim=-1)
    print("Selected blocks from baseline weights: ", blocks_selected)
    print("blocks selected shape: ", blocks_selected.shape)

    print("weight percentage met: ", weight_percentage_met)

if __name__ == '__main__':
    compare_attention(test_q, test_k)