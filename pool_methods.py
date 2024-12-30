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
    print("Query states shape: ", query_states.shape)
    B, H, L, D = query_states.shape
    num_chunks = L // block_size

    query_states, key_states = query_states.reshape(B * H, L, D), key_states.reshape(B * H, L, D)
    # print("Query and key states reshaped to: ", query_states.shape, key_states.shape)
    attn_weights = torch.einsum("bnd,bmd->bnm", query_states, key_states)
    # print("Attention weights: ", attn_weights.shape)
    print("Attention weights before applying causal mask: ", attn_weights[0, -1])
    attn_weights = apply_causal_mask(attn_weights)
    print("Unsoftmaxed original attention weights: ", attn_weights[0, -1])  # first head of first batch
    print("Softmaxing manually: ", attn_weights[0, -1].softmax(dim=-1))
    attn_weights = attn_weights.softmax(dim=-1)
    print("After softmax: ", attn_weights[0, -1])
    print("After softmax shape: ", attn_weights.shape)
    # print("Attention weights shape after softmax: ", attn_weights.shape)
    attn_weights = attn_weights.reshape(B * H, L, num_chunks, block_size)
    print("Attention weights values after reshape: ", attn_weights[0, -1])
    print("Attention weights shape after reshape: ", attn_weights.shape)
    attn_weights = attn_weights.sum(dim=-1) 
    print("Attention weights after the sum: ", attn_weights.shape)
    attn_weights = attn_weights.reshape(B * H, L, num_chunks)
    print("Attention weights after summing: ", attn_weights[0, -1])
    attn_weights = torch.log(attn_weights)
    print("Attention weights after logging: ", attn_weights[0, -1])
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

def apply_causal_mask(attn_weights):
    B, L, _ = attn_weights.shape
    mask = torch.triu(torch.ones(L, L), diagonal=1).to(attn_weights.device)
    mask = mask.unsqueeze(0)
    mask = mask.repeat(B, 1, 1)
    mask = mask.to(attn_weights.device)
    return attn_weights.masked_fill(mask == 1, float('-inf'))

def compare_divergence(top_blocks_regular, top_blocks_generated, block_size):
    B, H, num_chunks, D = top_blocks_generated.shape
    B, H, T, D = top_blocks_regular.shape
    # print("Top blocks generated shape: ", top_blocks_generated.shape)
    # print("Top blocks regular: ", top_blocks_regular.shape)
    print("Starting top blocks generated shape: ", top_blocks_generated.shape)
    # top_blocks_generated = top_blocks_generated.unsqueeze(3)
    print("After unsqueeze: ", top_blocks_generated.shape)
    top_blocks_generated = top_blocks_generated.repeat(1, 1, 1, block_size)
    print("After repeat: ", top_blocks_generated.shape)
    top_blocks_generated = top_blocks_generated.reshape(B, H, num_chunks * block_size, num_chunks)
    print("Top blocks generated: ", top_blocks_generated.shape)

    top_blocks_regular = torch.where(
        top_blocks_regular == -float("inf"),
        torch.full_like(top_blocks_regular, -1e10),
        top_blocks_regular,
    )

    # top_blocks_generated = torch.where(
    #     top_blocks_generated == 0,
    #     torch.full_like(top_blocks_generated, 1e10),
    #     top_blocks_generated
    # )

    # print("Regular distribution for the first token: ", top_blocks_regular[0, 0, 0])
    # print("Generated distribution for the first token: ", top_blocks_generated[0, 0, 0])

    # print("Top blocks regular shape: ", top_blocks_regular.shape)
    # print("Top blocks generated shape: ", top_blocks_generated.shape)

    # for i in range(100):
    #     print(
    #         "Item values: Regular: ",
    #         top_blocks_regular[0, 0, i * 16],
    #         " Generated: ",
    #         top_blocks_generated[0, 0, i * 16],
    #         top_blocks_generated[0, 0, i * 16].sum(),
    #     )
    #     print(
    #         "KL divergence for token at index: ",
    #         i * 16,
    #         F.kl_div(
    #             top_blocks_regular[0, 0, i * 16],
    #             top_blocks_generated[0, 0, i * 16],
    #             reduction="mean",
    #         )
    #     )

    # print("G reshaped shape: ", g_reshaped.shape)
    top_blocks_regular = top_blocks_regular.reshape(-1, num_chunks)
    top_blocks_generated = top_blocks_generated.reshape(-1, num_chunks)

    print("Regular shape: ", top_blocks_regular.shape, " Generated shape: ", top_blocks_generated.shape)
    kl_divergences = torch.zeros((top_blocks_regular.shape[0]))
    for i in range(top_blocks_regular.shape[0]):
        kl_divergences[i] = F.kl_div(top_blocks_regular[i], top_blocks_generated[i])

    # T - 1 gives us the (0, 1) index we were using for debugging before
    print(
        "Regular distribution: ",
        top_blocks_regular[T - 1],
        "\n Exponential of regular distribution: ",
        torch.exp(top_blocks_regular[T - 1]),
        "\nSum of regular distribution exponential: ",
        torch.exp(top_blocks_regular[T - 1]).sum(),
        "\nGenerated blocks: ",
        top_blocks_generated[T-1],
        "\nGenerated blocks sum: ",
        top_blocks_generated[T-1].sum(),
        "\nKL Divergence: ",
        kl_divergences[T-1],
    )
    print("KL Divergence Maximum Value: ", torch.max(kl_divergences))
    max_kldiv_index = torch.argmax(kl_divergences)
    print("KL div index: ", max_kldiv_index, max_kldiv_index / (B * H), max_kldiv_index % (B * H))
    print("Top blocks regular: ", top_blocks_regular[max_kldiv_index])
    print("Exp: ", torch.exp(top_blocks_regular[max_kldiv_index]))
    print("Top blocks generated", top_blocks_generated[max_kldiv_index])
    print("KL Divergence Mean: ", torch.mean(kl_divergences))

    kl_div = F.kl_div(top_blocks_regular, top_blocks_generated, reduction="batchmean")
    print("KL Divergence: ", kl_div)
    return kl_div.item()

def avg_pooling(query_states, key_states, block_size):
    B, H, T, D = query_states.shape
    num_chunks = T // block_size
    query_states, key_states = reshape_qkv(query_states, key_states, block_size)
    query_states, key_states = query_states.mean(dim=-2), key_states.mean(dim=-2)
    print("QK shape: ", query_states.shape, key_states.shape)
    attn_weights = torch.einsum("bnd,bmd->bnm", query_states, query_states)
    print("Attention weights shape: ", attn_weights.shape)
    attn_weights = apply_causal_mask(attn_weights)
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
    attn_weights = apply_causal_mask(attn_weights)
    attn_weights = attn_weights.softmax(dim=-1)
    attn_weights = attn_weights.reshape(B, H, num_chunks, num_chunks)
    return attn_weights

def max_pooling(query_states, key_states, block_size):
    B, H, T, D = query_states.shape
    num_chunks = T // block_size
    query_states, key_states = reshape_qkv(query_states, key_states, block_size)
    query_states, _ = query_states.max(dim=-2)
    key_states, _ = key_states.max(dim=-2)
    attn_weights = torch.einsum("bnd,bmd->bnm", query_states, query_states)
    attn_weights = apply_causal_mask(attn_weights)
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
    attn_weights = apply_causal_mask(attn_weights)
    attn_weights = attn_weights.softmax(dim=-1)
    attn_weights = attn_weights.reshape(B, H, num_chunks, num_chunks)
    return attn_weights

def max_softmax_pooling(query_states, key_states, block_size):
    B, H, T, _ = query_states.shape
    num_chunks = T // block_size
    query_states, key_states = reshape_qkv(query_states, key_states, block_size)
    query_states, _ = query_states.max(dim=-2)
    key_states, _ = key_states.max(dim=-2)
    # print("Query and key states shape: ", query_states.shape, key_states.shape)
    query_states, key_states = query_states.softmax(dim=-1), key_states.softmax(dim=-1)
    # attn_weights = torch.einsum("bnd,bmd->bnm", query_states, query_states)
    attn_weights = apply_causal_mask(attn_weights)
    attn_weights = attn_weights.softmax(dim=-1)
    attn_weights = attn_weights.reshape(B, H, num_chunks, num_chunks)
    return attn_weights

pooling_methods = {
    "avg_pooling": avg_pooling,
    # "max_softmax_pooling": max_softmax_pooling,
    # "softmax_max_pooling": softmax_max_pooling,
    # "softmax_avg_pooling": softmax_avg_pooling,
    # "max_pooling": max_pooling
}
