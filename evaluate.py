from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
from dataclasses import dataclass
from torch import nn
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, LlamaAttention, LlamaSdpaAttention
from transformers.cache_utils import Cache
import math
from typing import List
from pool_methods import pooling_methods, baseline_pooling, compare_divergence

@dataclass
class Result():
    method_name: str
    divergence: float
    block_size: int
    layer_index: int

# model_name = "unsloth/Llama-3.2-1B-Instruct" # for when the regular model is inaccessible
model_name = "meta-llama/Llama-3.2-1B-Instruct"
dataset_name = "abacusai/LongChat-Lines"
block_size = 64

results: List[Result] = []

# @torch.compile
def custom_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        raise NotImplementedError()

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    regular_blocks = baseline_pooling(query_states, key_states, block_size)
    for method in pooling_methods:
        method_blocks = pooling_methods[method](query_states, key_states, block_size)
        divergence = compare_divergence(regular_blocks, method_blocks, block_size)
        results.append(Result(method, divergence, block_size, self.layer_idx))

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

LlamaSdpaAttention.forward = custom_forward

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa").to(device) 
print("Finished loading model")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
ds = load_dataset(dataset_name, split="250")
ds = ds.select(list(range(8)))

prompts = ds["prompt"]

for prompt in prompts:
    inputs = tokenizer([prompt], return_tensors="pt", max_length=4096, padding="max_length", truncation=True).to(device)
    outputs = model(**inputs)

with open(f"bsa_results_{block_size}.json", "w+") as f:
    f.write(json.dumps([result.__dict__ for result in results], indent=2))
