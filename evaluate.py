from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import json
from tqdm import tqdm
from dataclasses import dataclass
from torch import nn
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, LlamaAttention
from transformers.cache_utils import Cache
import math
from typing import List

@dataclass
class Result():
    method_name: str
    divergence: float
    block_size: int
    layer_index: int

model_name = "meta-llama/Llama-3.2-1B"
dataset_name = "abacusai/LongChat-Lines"
block_size = 64

results: List[Result] = []

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
    global results
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
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    print("Shapes: ", query_states.shape, key_states.shape, attn_weights.shape)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # use this
    # regular_blocks = get_top_blocks_regular(query_states, key_states, block_size)
    # for method in pooling_methods:
    #     method_blocks = pooling_methods[method](query_states, key_states, block_size)
    #     divergence = compare_divergence(regular_blocks, method_blocks, block_size)
    #     results.append(Result(method, divergence, block_size, self.layer_idx))
    
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

LlamaAttention.forward = custom_forward

model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager") 
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
ds = load_dataset(dataset_name, split="950")
ds = ds.select(list(range(8)))

prompts = ds["prompt"]

num_prompts = len(prompts)
inputs = tokenizer(prompts, return_tensors="pt", max_length=4096, padding="max_length", truncation=True)
outputs = model(**inputs)
    
with open(f"bsa_results_{block_size}.json", "w+") as f:
    f.write(json.dumps([result.__dict__ for result in results], indent=2))