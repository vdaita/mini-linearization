import torch
import os
from rich import print
# Get terminal width
terminal_width = os.get_terminal_size().columns
torch.set_printoptions(linewidth=terminal_width, sci_mode=False, precision=4)
torch.manual_seed(42)


def apply_causal_mask(attn_weights):
    B, L, _ = attn_weights.shape
    mask = torch.triu(torch.ones(L, L), diagonal=1).to(attn_weights.device)
    mask = mask.unsqueeze(0)
    mask = mask.repeat(B, 1, 1)
    mask = mask.to(attn_weights.device)
    return attn_weights.masked_fill(mask == 1, float('-inf'))

def baseline_attention(q, k, block_size):
    # print("=====Running Baseline Attention=====")
    B, H, T, D = q.shape
    num_chunks = T // block_size
    q, k = q.reshape(B * H, T, D), k.reshape(B * H, T, D)
    # print("QK shape: ", q.shape, k.shape)
    attn_weights = torch.einsum('bnd,bmd->bnm', q, k)
    attn_weights = apply_causal_mask(attn_weights)
    attn_weights = torch.softmax(attn_weights, dim=-1)

    attn_weights = attn_weights.reshape(B * H, T, num_chunks, block_size)
    attn_weights = attn_weights.sum(dim=-1)

    baseline_diag_fill_indices_x = torch.arange(T)
    baseline_diag_fill_indices_y = torch.arange(num_chunks).repeat_interleave(block_size)
    attn_weights[:, baseline_diag_fill_indices_x, baseline_diag_fill_indices_y] = 0
    
    denominator = attn_weights.sum(dim=-1, keepdim=True)
    denominator = torch.where(denominator == 0, torch.tensor(1.0, device=attn_weights.device), denominator)
    attn_weights = attn_weights / denominator
    # print("Attn weights shape: ", attn_weights.shape)
    return attn_weights

def reshape_qk(q, k, block_size):
    B, H, T, D = q.shape
    q, k = q.reshape(B * H, T, D), k.reshape(B * H, T, D)
    q, k = q.reshape(B * H, T // block_size, block_size, D), k.reshape(B * H, T // block_size, block_size, D)
    return q, k

def averaged_attention(q, k, block_size):
    # print("=====Running Averaged Attention=====")
    # print("QK shape: ", q.shape, k.shape)
    q, k = reshape_qk(q, k, block_size)
    q, k = q.mean(dim=2), k.mean(dim=2)
    attn_weights = torch.einsum('bnd,bmd->bnm', q, k)
    attn_weights = apply_causal_mask(attn_weights)
    attn_weights = torch.softmax(attn_weights, dim=-1)
    # print("Attn weights shape: ", attn_weights.shape)
    return attn_weights

def softmax_averaged_attention(q, k, block_size):
    q, k = reshape_qk(q, k, block_size)
    q, k = q.softmax(dim=-1), k.softmax(dim=-1)
    q, k = q.mean(dim=-2), k.mean(dim=-2)
    attn_weights = torch.einsum("bnd,bmd->bnm", q, k)
    attn_weights = apply_causal_mask(attn_weights)
    attn_weights = attn_weights.softmax(dim=-1)
    return attn_weights

def max_attention(q, k, block_size):
    q, k = reshape_qk(q, k, block_size)
    q, _ = q.max(dim=-2)
    k, _ = k.max(dim=-2)
    attn_weights = torch.einsum("bnd,bmd->bnm", q, k)
    attn_weights = apply_causal_mask(attn_weights)
    attn_weights = attn_weights.softmax(dim=-1)
    return attn_weights

def softmax_max_attention(q, k, block_size):
    q, k = reshape_qk(q, k, block_size)
    q, k = q.softmax(dim=-1), k.softmax(dim=-1)
    q, _ = q.max(dim=-2)
    k, _ = k.max(dim=-2)
    attn_weights = torch.einsum("bnd,bmd->bnm", q, k)
    attn_weights = apply_causal_mask(attn_weights)
    attn_weights = attn_weights.softmax(dim=-1)
    return attn_weights

def max_min_attention(q, k, block_size):
    q, k = reshape_qk(q, k, block_size)
    q = torch.cat([q.max(dim=-2)[0], q.min(dim=-2)[0]], dim=-1)
    k = torch.cat([k.max(dim=-2)[0], k.min(dim=-2)[0]], dim=-1)
    attn_weights = torch.einsum("bnd,bmd->bnm", q, k)
    attn_weights = apply_causal_mask(attn_weights)
    attn_weights = attn_weights.softmax(dim=-1)
    return attn_weights

def softmax_max_min_attention(q, k, block_size):
    q, k = reshape_qk(q, k, block_size)
    q, k = q.softmax(dim=-1), k.softmax(dim=-1)
    q = torch.cat([q.max(dim=-2)[0], q.min(dim=-2)[0]], dim=-1)
    k = torch.cat([k.max(dim=-2)[0], k.min(dim=-2)[0]], dim=-1)
    attn_weights = torch.einsum("bnd,bmd->bnm", q, k)
    attn_weights = apply_causal_mask(attn_weights)
    attn_weights = attn_weights.softmax(dim=-1)
    return attn_weights

methods = {
    "averaged": averaged_attention,
    "softmax_averaged": softmax_averaged_attention,
    "max": max_attention,
    "softmax_max": softmax_max_attention,
    "max_min": max_min_attention,
    "softmax_max_min": softmax_max_min_attention
}

# @torch.compile
def comparison_score(baseline_weights, averaged_weights, q, block_size):
    B, H, T, D = q.shape
    num_chunks = T // block_size

    num_selected = num_chunks
    averaged_diag_fill_indices_x = torch.arange(num_chunks)
    averaged_diag_fill_indices_y = torch.arange(num_chunks)
    averaged_weights[:, averaged_diag_fill_indices_x, averaged_diag_fill_indices_y] = 0
    top_averaged_blocks = torch.topk(averaged_weights, num_selected, dim=-1).indices
    # print("Top averaged blocks before repeat: ", top_averaged_blocks)
    top_averaged_blocks = torch.repeat_interleave(top_averaged_blocks, block_size, dim=-2)
    # print("Top averaged blocks: ", top_averaged_blocks)

    # Use advanced indexing to parallelize
    i = torch.arange(B * H).unsqueeze(1).unsqueeze(2)  # Shape: (B * H, 1, 1)
    j = torch.arange(T).unsqueeze(0).unsqueeze(2)    # Shape: (1, T, 1)

    # Gather selected blocks based on indices
    blocks_selected = baseline_weights[i, j, top_averaged_blocks]

    weight_percentage_met = blocks_selected.cumsum(dim=-1)
    weight_percentage_met = weight_percentage_met[:, block_size:, :]

    # print("Selected blocks from baseline weights: ", blocks_selected)
    # print("blocks selected shape: ", blocks_selected.shape)

    # print("weight percentage met: ", weight_percentage_met)
    # print("weight percentage met shape: ", weight_percentage_met.shape)

    # now, the last dimension of weight_percentage_met is increasing the more blocks you add
    # that means to take the average, you have to average over the first and second dimensions

    # average over the first and second dimensions
    weight_percentage_met = weight_percentage_met.mean(dim=-2)
    # print("Mean weight percentage met: ", weight_percentage_met, " with shape: ", weight_percentage_met.shape)
    weight_percentage_met = weight_percentage_met.mean(dim=0)
    # print("Overall mean weight percentage met: ", weight_percentage_met, " with shape: ", weight_percentage_met.shape)

    # print("Weight percentage met: ", weight_percentage_met)
    return weight_percentage_met


@torch.compile
def evaluate_methods(q, k, block_size):
    B, H, T, D = q.shape
    num_chunks = T // block_size
    baseline_weights = baseline_attention(q, k, block_size)

    # print("baseline weights: ", baseline_weights)
    # print("baseline weights shape: ", baseline_weights.shape)
    # print("baseline weights with mean: ", baseline_weights)
    # print("averaged weights: ", averaged_weights)
    
    results = {
    }

    for method in methods:
        # print(f"Running {method} method")
        method_weights = methods[method](q, k, block_size)
        results[method] = comparison_score(baseline_weights, method_weights, q, block_size)

    return results

if __name__ == '__main__':
    B = 1
    H = 32
    T = 64 * 256
    D = 64

    block_size = 64

    test_q = torch.randn(B, H, T, D)
    test_k = torch.randn(B, H, T, D)
    results = evaluate_methods(test_q, test_k, block_size)
    print(results)
