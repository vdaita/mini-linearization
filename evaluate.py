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
from pool_methods import evaluate_methods, methods
from tqdm import tqdm

# model_name = "unsloth/Llama-3.2-1B-Instruct" # for when the regular model is inaccessible
model_name = "meta-llama/Llama-3.2-1B-Instruct"
block_size = 64
results = {
    method: torch.empty(0) for method in methods
}
num_example_chunks_processed = 0
dataset_name = "THUDM/LongBench-v2"
head_chunk_size = 1

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
    global results, num_example_chunks_processed
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

    for chunk_start in range(0, bsz, head_chunk_size):
        query_states_segment = query_states[:, chunk_start:chunk_start + head_chunk_size, ...]
        key_states_segment = key_states[:, chunk_start:chunk_start + head_chunk_size, ...]
        # print("Query: ", query_states_segment.shape, "Key: ", key_states_segment.shape)
        batch_results = evaluate_methods(query_states_segment, key_states_segment, block_size)
        for method in batch_results:
            if num_example_chunks_processed == 0:
                results[method] = batch_results[method]
            else:
                results[method] = results[method] * (
                    num_example_chunks_processed / (num_example_chunks_processed + bsz)
                ) + batch_results[method] * (bsz / (num_example_chunks_processed + bsz))
        num_example_chunks_processed += 1
        torch.cuda.synchronize()

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
ds = load_dataset(dataset_name, split="train")
ds = ds.shuffle()
ds = ds.filter(lambda x: x["length"] == "long")
ds = ds.select(list(range(100)))
ds = ds.to_pandas()
# print(ds)
# num_tokens = 64 * 256
num_tokens = 64 * 500

torch.set_printoptions(sci_mode=False)
with torch.no_grad():
    for _, row in tqdm(ds.iterrows()):
        question = row["question"]
        choice_a = row["choice_A"]
        choice_b = row["choice_B"]
        choice_c = row["choice_C"]
        choice_d = row["choice_D"]

        question = f"Question: {question}\nChoice A: {choice_a}\nChoice B: {choice_b}\nChoice C: {choice_c}\nChoice D: {choice_d}\n"
        required_tokens = len(tokenizer.encode(question))

        document_text = row["context"]
        encoded_document = tokenizer.encode(document_text)
        truncated_document = tokenizer.decode(encoded_document[:(num_tokens - required_tokens)])
        prompt = f"{truncated_document}\n{question}"
        prompt = tokenizer.apply_chat_template([
            {
                "role": "system",
                "content": "You are an intelligent assistant.",
            },
            {
                "role": "user",
                "content": prompt
            }
        ], tokenize=False, add_generation_prompt=True)

        prompt_tokens_length = len(tokenizer.encode(prompt))
        # print("Prompt tokens length: ", prompt_tokens_length)
        if prompt_tokens_length < num_tokens:
            print("Prompt tokens length less than num tokens")
        
        inputs = tokenizer([prompt], return_tensors="pt", max_length=num_tokens, padding="max_length", truncation=True).to(device)       
        outputs = model(**inputs)
        torch.cuda.synchronize()

for result in results:
    results[result] = results[result].tolist()

with open(f"bsa_results_{block_size}.json", "w+") as f:
    f.write(json.dumps(results, indent=2))
