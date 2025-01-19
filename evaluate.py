from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from tqdm import tqdm
from dataclasses import dataclass
from torch import nn
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, Unpack, FlashAttentionKwargs, Callable, eager_attention_forward
from transformers.cache_utils import Cache
import torch.nn.functional as F
from attention_methods import DiffLinearAttentionWeights, LinearAttentionWeights, HedgehogFeatureMap, LinearFeatureMap

import wandb

model_name = "meta-llama/Llama-3.2-1B-Instruct"
dataset_name = "THUDM/LongBench-v2"
torch.set_printoptions(sci_mode=False)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa").to(device) 
dataset = load_dataset(dataset_name, split="train")
dataset = dataset.batch(batch_size=8)

print("Finished loading model")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
head_dim = model.config.hidden_size // model.config.num_attention_heads
n_heads = model.config.num_attention_heads

with torch.grad():
    methods = [
        [
            DiffLinearAttentionWeights(n_heads, head_dim, depth, "hedgehog", "hedgehog"),
            LinearAttentionWeights(n_heads, head_dim, "hedgehog"),
            DiffLinearAttentionWeights(n_heads, head_dim, depth, "linear", "linear"),
            LinearAttentionWeights(n_heads, head_dim, "linear"),
            DiffLinearAttentionWeights(n_heads, head_dim, depth, "hedgehog", "linear"),
        ] 
        for depth in range(model.config.num_hidden_layers)
    ]

    optimizers = [
        [    
            torch.optim.AdamW(method.parameters(), lr=1e-5)
            for method in methods
        ]
    ]
    
    def custom_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        for (method, optimizer) in zip(methods[self.layer_idx], optimizers[self.layer_idx]):
            generated_attention_weights = method(query_states, key_states, attention_mask)
            loss = F.mse_loss(attn_weights, generated_attention_weights)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    for layer in model.model.layers:
        layer.forward = custom_forward

    for batch in tqdm(dataset):
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=1024)
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        model(**inputs)