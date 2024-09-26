import os
import numpy as np

# Set the number of threads for OpenBLAS
os.environ["OPENBLAS_NUM_THREADS"] = "16"  # Set this to the number of threads you want
os.environ["OMP_NUM_THREADS"] = "16"  # Set this to the number of OpenMP threads you want

# Optionally, you can also control the behavior of MKL if it's installed
os.environ["MKL_NUM_THREADS"] = "16" 

from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers.models.llama import LlamaForCausalLM
import torch
import time

import time
import types

def timing(f):
    def wrap(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        
        # Determine if the function is a method of a class
        if isinstance(args[0], object) and hasattr(args[0], '__class__'):
            class_name = args[0].__class__.__name__
            print(f'Function {class_name}.{f.__name__} took {end_time - start_time:.4f} seconds')
        else:
            print(f'Function {f.__name__} took {end_time - start_time:.4f} seconds')
        
        return result
    return wrap







class LlamaRotaryEmbedding:
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (np.arange(0, self.dim, 2).astype(np.float32) / self.dim))

    def __call__(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = np.expand_dims(self.inv_freq, axis=0).astype(np.float32)
        inv_freq_expanded = np.expand_dims(inv_freq_expanded, axis=-1).repeat(position_ids.shape[0], axis=0)
        position_ids_expanded = np.expand_dims(position_ids, axis=1).astype(np.float32)
        
        freqs = np.matmul(inv_freq_expanded, position_ids_expanded).transpose(0, 2, 1)
        emb = np.concatenate((freqs, freqs), axis=-1)
        cos = np.cos(emb)
        sin = np.sin(emb)
        
        return cos.astype(x.dtype), sin.astype(x.dtype)








def rotate_half_np(x):
    """
    Rotates half the hidden dims of the input, implemented in NumPy.
    Args:
        x (numpy.ndarray): Input array.
    Returns:
        numpy.ndarray: Rotated array.
    """
    # Split the array into two halves along the last dimension
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    # Reverse the second half and concatenate with the first half
    # NumPy's negative indexing reverses the array
    return np.concatenate((-x2, x1), axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = np.expand_dims( cos, unsqueeze_dim)
    sin = np.expand_dims( sin, unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half_np(q) * sin)
    k_embed = (k * cos) + (rotate_half_np(k) * sin)
    return q_embed, k_embed


import math

#timing
def gelu_np(input):
    return 0.5 * input * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * np.power(input, 3.0))))



def silu(x):
    # Sigmoid function (σ(x))
    sigmoid_x = 1 / (1 + np.exp(-x))
    # SiLU function: x * sigmoid(x)
    return x * sigmoid_x




# Example activation function mapping (adjust as needed)
ACT2FN_np = {
    'relu': np.vectorize(lambda x: max(0, x)),
    # Add other activation functions as needed
    'gelu_pytorch_tanh': gelu_np,
    'silu': silu
}







class Linear_np:
    def __init__(self, in_features, out_features, bias=True):
        # Initialize weights and bias
        self.weights = None
        self.bias = None

    def load(self, weights, bias=None):
        self.weights = weights
        self.bias = bias

    def __call__(self, x):
        # Preallocate the output array to avoid creating new arrays repeatedly
        out = np.empty((*x.shape[:-1], self.weights.shape[0]), dtype=x.dtype)
        
        # Perform the linear operation (y = xA^T + b)
        np.dot(x, self.weights.T, out=out)
        
        if self.bias is not None:
            out += self.bias  # This uses broadcasting, which is efficient in numpy
        
        return out









class LlamaMLP_np:
    def __init__(self, config, layer_index):
        self.config = config
        self.layer_index = layer_index
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj_weights = Linear_np(self.hidden_size, self.intermediate_size)
        self.up_proj_weights = Linear_np(self.hidden_size, self.intermediate_size)
        self.down_proj_weights = Linear_np(self.intermediate_size, self.hidden_size)

        gate_proj_weights= load_weights(f'model.layers.{layer_index}.mlp.gate_proj.weight')

        down_proj_weights= load_weights(f'model.layers.{layer_index}.mlp.down_proj.weight')

        up_proj_weights= load_weights(f'model.layers.{layer_index}.mlp.up_proj.weight')

        self.gate_proj_weights.load(gate_proj_weights)
        self.down_proj_weights.load(down_proj_weights)
        self.up_proj_weights.load(up_proj_weights)

        self.act_fn = ACT2FN_np[config.hidden_act]

    def __call__(self, x):
        gate_proj_output = self.gate_proj_weights(x)
        up_proj_output = self.up_proj_weights(x)
        activated_output = self.act_fn(gate_proj_output)
        multiplied_output = activated_output * up_proj_output
        down_proj_output = self.down_proj_weights(multiplied_output)
        return down_proj_output




#timing
def repeat_kv_np(hidden_states, n_rep):
    """
    Replicates the behavior of torch.repeat_interleave for a specific use-case. 
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) 
    to (batch, num_attention_heads, seqlen, head_dim) in NumPy.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    
    if n_rep == 1:
        return hidden_states
    
    # Expand and then repeat the array along the third axis
    hidden_states_expanded = np.expand_dims(hidden_states, axis=2)
    hidden_states_repeated = np.tile(hidden_states_expanded, (1, 1, n_rep, 1, 1))

    # Reshape the array to the desired shape
    return hidden_states_repeated.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# # Example usage
# hidden_states = np.random.randn(2, 3, 4, 5)  # Example input tensor with shape (batch, num_key_value_heads, slen, head_dim)
# n_rep = 2  # Example repetition factor
# output = repeat_kv_np(hidden_states, n_rep)
# print(output.shape)  # The shape should be (2, 6, 4, 5) in this example

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# my_dict = {'id': 1, 'website': 'bobbyhadz.com', 'topic': 'Python'}

# new_dict = AttributeDict(my_dict)

# print(new_dict.website)  # 👉️ bobbyhadz.com
# print(new_dict.topic)  # 👉️ Python

# new_dict.author = 'Borislav Hadzhiev'

# # 👇️ {'id': 1, 'website': 'bobbyhadz.com', 'topic': 'Python', 'author': 'Borislav Hadzhiev'}
# print(new_dict)

# del new_dict.author

# # 👇️ {'id': 1, 'website': 'bobbyhadz.com', 'topic': 'Python'}
# print(new_dict)

# config1 = AttributeDict(config)
# config1.vocab_size








class LlamaRMSNorm_np:
    def __init__(self, dim: int,layer_idx =-1, eps: float = 1e-6, loc='input_layernorm'):
        self.eps = eps
        if layer_idx == -1:
        
            gamma = load_weights(f'model.norm.weight')            
        else:
            gamma = load_weights(f'model.layers.{layer_idx}.{loc}.weight')

        self.weight = gamma

    def _norm(self, x):
        return x * np.reciprocal(np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + self.eps))
    # def _norm(self, x):
    #     # Cast x to float32 for the computation to avoid overflow
    #     x = x.astype(np.float32)
        
    #     # Compute the mean square and add eps to prevent division by zero
    #     mean_square = np.mean(np.square(x), axis=-1, keepdims=True)
        
    #     # Calculate the reciprocal of the root mean square
    #     norm_factor = np.reciprocal(np.sqrt(np.maximum(mean_square, self.eps)))
        
    #     # Apply the normalization factor
    #     normalized_x = x * norm_factor
        
    #     # Cast back to float16 if needed
    #     return normalized_x.astype(np.float32)


    def __call__(self, x):
        output = self._norm(x)
        output = output * self.weight
        return output

    def extra_repr(self):
        return f"{self.weight.shape}, eps={self.eps}"




def softmax1(x, axis=-1):
    """Compute softmax values for each sets of scores in x along the specified axis."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def softmax(x, axis=-1):
    """Compute softmax values for each sets of scores in x along the specified axis."""
    e_x = np.exp(x )
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def softmax2(x, axis=-1):
    """Compute softmax values for each sets of scores in x along the specified axis."""
    e_x = np.exp(x )
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# # Example usage
# attn_weights_np = np.random.randn(2, 4, 10, 10).astype(np.float32)  # Example attention weights
# softmax_attn_weights_np = softmax(attn_weights_np, axis=-1)

# If you need to convert back to a specific dtype, like in PyTorch code, you can cast it:
# e.g., softmax_attn_weights_np = softmax_attn_weights_np.astype(original_dtype)


from typing import Tuple

class KVCache:
    def __init__(self):
        self.key_cache = []
        self.value_cache = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            return self.key_cache[0].shape[-2]

    #timing
    def update(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        layer_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(keys)
            self.value_cache.append(values)
        else:
            self.key_cache[layer_idx] = np.concatenate(
                [self.key_cache[layer_idx], keys], axis=-2
            )
            self.value_cache[layer_idx] = np.concatenate(
                [self.value_cache[layer_idx], values], axis=-2
            )
            
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

class LlamaAttention():
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx = None):
        
        self.config = config
        self.layer_idx = layer_idx
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        
        
        if (self.hidden_size % self.num_heads) != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
    

        self._init_rope()

        self.q_proj = Linear_np(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = Linear_np(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = Linear_np(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = Linear_np(self.num_heads * self.head_dim, self.hidden_size, bias=True)
        
        
        
        q_proj =load_weights(f'model.layers.{layer_idx}.self_attn.q_proj.weight')
        k_proj =load_weights(f'model.layers.{layer_idx}.self_attn.k_proj.weight')
        v_proj =load_weights(f'model.layers.{layer_idx}.self_attn.v_proj.weight')
        o_proj = load_weights(f'model.layers.{layer_idx}.self_attn.o_proj.weight')

        self.q_proj.load(q_proj)
        self.k_proj.load(k_proj)
        self.v_proj.load(v_proj)
        self.o_proj.load(o_proj)


    def _init_rope(self):
        # self.rotary_emb = LlamaRotaryEmbedding(
        #     self.config
        # )
        self.rotary_emb = LlamaRotaryEmbedding(
            int(self.head_dim),
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        
    
    def _split_heads(self, fused_qkv):
        batch_size, seq_length, _ = fused_qkv.shape
        reshaped = fused_qkv.reshape(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        query = reshaped[..., 0, :]
        key = reshaped[..., 1, :]
        value = reshaped[..., 2, :]
        return query, key, value
        
    def __call__(
        self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        output_attentions = False,
        use_cache = False,
        kv_cache = None,
        position_embeddings = None
        
    ): 
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)

        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)
        
        kv_seq_len = key_states.shape[-2]
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin= position_embeddings

        # # Partial rotary embedding
        # query_rot, query_pass = (
        #     query_states[..., : self.rotary_emb.dim],
        #     query_states[..., self.rotary_emb.dim :],
        # )
        # key_rot, key_pass = (
        #     key_states[..., : self.rotary_emb.dim],
        #     key_states[..., self.rotary_emb.dim :],
        # )
        # # [batch_size, seq_length, num_heads, head_dim // config.partial_rotary_factor]
        # query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)

        # # [batch_size, seq_length, num_heads, head_dim]
        # query_states = np.concatenate((query_rot, query_pass), axis=-1)
        
        # key_states = np.concatenate((key_rot, key_pass), axis=-1)

        # print(query_states.shape, key_states.shape)

        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # print(query_states.shape, key_states.shape)

        #print(self.layer_idx, "before", kv_cache.num_items())

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        #print(self.layer_idx,"after", kv_cache.num_items())
        
        
        key_states = repeat_kv_np(key_states, self.num_key_value_groups)
        value_states = repeat_kv_np(value_states, self.num_key_value_groups)

        
        # Queries and keys upcast to fp32 is required by Phi-2 to avoid overflow
        attn_weights = np.matmul(
            query_states, key_states.transpose(0, 1, 3, 2)
        ) / np.sqrt(self.head_dim)

        if q_len > 2:

            tril = np.tril(np.ones((q_len, q_len), dtype=np.float32))
            
            
            mask = tril[None, None, :, :]  # Add a new dimension at the beginning to match batch size
            
            attn_weights = np.where(mask == 0, float('-inf'), attn_weights)
        




        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        
        attn_weights = softmax(attn_weights, axis=-1)
        
        

        attn_output = np.matmul(attn_weights, value_states)
        
        

        attn_output = attn_output.transpose(0, 2, 1, 3)
        
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, kv_cache


class LlamaDecoderLayer():
    def __init__(self, config, layer_idx: int):
        self.self_attn = LlamaAttention(config, layer_idx=layer_idx)
        self.mlp = LlamaMLP_np(config, layer_index=layer_idx)
        self.input_layernorm = LlamaRMSNorm_np(config.hidden_size, layer_idx=layer_idx, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm_np(config.hidden_size, layer_idx=layer_idx, eps=config.rms_norm_eps, loc='post_attention_layernorm')
       
    def __call__(
        self,
        hidden_states,
        attention_mask = None,
        position_ids  = None,
        output_attentions  = False,
        use_cache  = False,
        kv_cache = None,
        position_embeddings = None
    ) :
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, kv_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=use_cache,
            kv_cache=kv_cache,
            position_embeddings= position_embeddings
        )
        
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        

        
        
        hidden_states = self.mlp(hidden_states)
       
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)

        outputs += (self_attn_weights,)

        outputs += (kv_cache,)

        return outputs

class LlamaModel():


    def __init__(self, config):
        
        self.config = config
        #self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = load_weights('model.embed_tokens.weight')
        
        self.layers = [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        
        self.final_layernorm = LlamaRMSNorm_np(config.hidden_size, eps=config.rms_norm_eps)
        self._use_flash_attention_2 = False

        self.gradient_checkpointing = False
        self.head_dim = config.head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        #self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.rotary_emb = LlamaRotaryEmbedding(
            int(self.head_dim),
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        # Initialize weights and apply final processing
        

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    
    def __call__(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        inputs_embeds= None,
        use_cache= None,
        output_attentions = True,
        output_hidden_states= True,
        return_dict = None,
        kv_cache = None,
    ) :
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = False

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0

       


        if position_ids is None:
            
            position_ids = np.arange(
                0, seq_length, dtype=np.int64
            )
            
            if use_cache and  kv_cache is not None:
            
                position_ids = np.arange(
                        kv_cache.num_items(), seq_length + kv_cache.num_items(), dtype=np.int64
                    )
            

            position_ids = np.expand_dims(position_ids, 0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens[input_ids]


        
        attention_mask = None
        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

    

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        

        
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                use_cache=use_cache,
                kv_cache=kv_cache,
                position_embeddings= position_embeddings
            )

            hidden_states = layer_outputs[0]
            

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        
        last_hidden_state=hidden_states
        
        hidden_states=all_hidden_states
        attentions=all_self_attns
        return (
            last_hidden_state,
            kv_cache,
            hidden_states,
            attentions
        )

class LlamaForCausalLM_np():
    

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__ with Llama->Phi,bias=False->bias=True
    def __init__(self, config):
        self.config = config
        
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = Linear_np(config.hidden_size, config.vocab_size, bias=True)
        
        weights =load_weights(f'lm_head.weight')
        
        
        self.lm_head.load(weights)

        

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_input_embeddings
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_input_embeddings
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_output_embeddings
    def get_output_embeddings(self):
        return self.lm_head

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_decoder
    def set_decoder(self, decoder):
        self.model = decoder

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_decoder
    def get_decoder(self):
        return self.model

    
    def __call__(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = True,
        output_hidden_states= True,
        return_dict = True,
        kv_cache=None
    ) :
       
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = False #return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            kv_cache=kv_cache
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        #logits = softmax(logits, axis=-1)
        logits = logits
        


        loss = None

        logits=logits
        kv_cache=outputs[1]
        hidden_states=outputs[2]
        attentions=outputs[3]

        return (
            loss,
            logits,
            kv_cache,
            hidden_states,
            attentions,
        )

    
    
import random

def sample(probabilities, n):
    probabilities = softmax(probabilities)
    try:
    # Sample index from probabilities, they must sum to 1
        r = random.random()
        probabilities = probabilities
        cdf = 0.0
        for i in range(n):
            cdf += probabilities[i]
            if r < cdf:
                return i
        return n - 1  # In case of rounding errors
    except:
        print(probabilities)

# Example usage
# probabilities = np.array([0.1, 0.2, 0.3, 0.4])  # Example list of probabilities
# n = len(probabilities)  # Number of probabilities
# index = sample(probabilities, n)
# print(index)


import torch

def min_p_sampling(logits, p_base: float = 0.1) -> np.ndarray:
    logits = softmax(logits)
    p_max = np.max(logits, axis=-1, keepdims=True)
    p_scaled = p_max * p_base
    mask = logits >= p_scaled
    logits = logits * mask.astype(np.float32)
    logits = logits / logits.sum(axis=-1, keepdims=True)
    
    # Use numpy's version of multinomial sampling
    next_token =  torch.multinomial(torch.tensor(logits), num_samples=1)
    
    return next_token

def generate(prompt, tokenizer, model, max_tokens=20, streamer=None, kv_cache=None, config=None):
    inp = tokenizer.encode(prompt, add_special_tokens=False )
    

    past_key_values = None
    res = prompt
    print(prompt, end="", flush=True)

    for i in range(max_tokens):
        if config.use_cache or kv_cache is not None:
            if i ==0 :
                input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.numpy()
            else:
                input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.numpy()
        else:
            input_ids = tokenizer(res, return_tensors="pt", add_special_tokens=True).input_ids.numpy()
        
       
        input_ids = np.asarray(input_ids)
        loss, logits, kv_cache, hidden_states, attentions = model(input_ids,
                                                            use_cache = config.use_cache,
                                                            kv_cache = kv_cache
    )



        val = min_p_sampling(logits[-1][-1])
        #print(logits.shape)
        
        #val = sample(logits[-1][-1], vocab_size)
        # val = np.argmax(softmax(logits[-1][-1]), axis=-1)
        # val = np.asnumpy(val)


        prompt = tokenizer.decode(val)
        res += prompt
        print(prompt, end="", flush=True)
    return res




def softmax(x, axis=-1):
    
    e_x = np.exp(x )
    y = e_x / np.sum(e_x, axis=axis, keepdims=True)
    return y

# Example usage
attn_weights_np = np.random.randn(2, 4, 10, 10).astype(np.float32)  # Example attention weights
softmax_attn_weights_np = softmax(attn_weights_np, axis=-1)

# If you need to convert back to a specific dtype, like in PyTorch code, you can cast it:
# e.g., softmax_attn_weights_np = softmax_attn_weights_np.astype(original_dtype)





import torch.nn.functional as F

def softmax1(x, axis=-1):
    """Compute softmax values for each sets of scores in x along the specified axis."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def softmax2(x, axis=-1):
    """Compute softmax values for each sets of scores in x along the specified axis."""
    e_x = np.exp(x )
    return e_x / np.sum(e_x, axis=axis, keepdims=True)



import torch

def min_p_sampling(logits, p_base: float = 0.1) -> np.ndarray:
    
    logits =  softmax2(logits)
    
    p_max = np.max(logits, axis=-1, keepdims=True)
    p_scaled = p_max * p_base
    mask = logits >= p_scaled
    logits = logits * mask.astype(np.float32)
    logits = logits / logits.sum(axis=-1, keepdims=True)
        
    # Use numpy's version of multinomial sampling
    next_token =  torch.multinomial(torch.tensor(logits), num_samples=1)
    
    return next_token


# Example usage
# attn_weights_np = np.random.randn(2, 4, 10, 10).astype(np.float32)  # Example attention weights
# softmax_attn_weights_np = softmax(attn_weights_np, axis=-1)
# print(softmax_attn_weights_np)

# If you need to convert back to a specific dtype, like in PyTorch code, you can cast it:
# e.g., softmax_attn_weights_np = softmax_attn_weights_np.astype(original_dtype)

from huggingface_hub import notebook_login
notebook_login()
import os
import json
from safetensors import safe_open

from safetensors.torch import load_file


def load_sharded_safetensors_via_weight_map(model_dir, index_filename='model.safetensors.index.json'):
    """
    Loads tensors from sharded safetensors files based on the weight_map in the index file.

    Args:
        model_dir (str): Path to the model directory containing safetensors files.
        index_filename (str): Name of the index JSON file.

    Returns:
        dict: A dictionary mapping tensor names to CuPy arrays.
    """
    all_tensors = {} 
    try:
        index_path = os.path.join(model_dir, index_filename)
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        weight_map = index.get('weight_map', {})
        if not weight_map:
            raise ValueError("The weight_map is missing or empty in the index file.")
        
        # Group tensor names by their shard files
        shard_to_tensors = {}
        for tensor_name, shard_file in weight_map.items():
            shard_to_tensors.setdefault(shard_file, []).append(tensor_name)

        
        for shard_file in shard_to_tensors.keys():
            state_dict = load_file(os.path.join(model_dir, shard_file))
            all_tensors.update(state_dict)
    except:
        state_dict = load_file(os.path.join(model_dir, 'model.safetensors'))
        all_tensors.update(state_dict)

    
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("All shards loaded and tensors extracted successfully.")
    return all_tensors, AttributeDict(config)

weights = dict()
def load_weights(key):
    if key == 'lm_head.weight':
        key = 'model.embed_tokens.weight'
    t = weights[key].to(torch.float32)
    return t.numpy()

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=model_name )
    # Define the path to your model directory
    model_dir = path  # Replace with your actual path

    # Load all tensors from the sharded safetensors files
    global weights
    weights, config = load_sharded_safetensors_via_weight_map(model_dir)
    
    model = LlamaForCausalLM_np(config)
    return tokenizer, model, config

if __name__ == '__main__':
    tokenizer, model, config = load_model("meta-llama/Llama-3.2-1B")

    import time
    s = time.time()
    config.use_cache = True
    generate('Once upon a time',tokenizer, model, max_tokens=200,kv_cache=KVCache(), config=config)
    e = time.time()
    #print('\n',e-s)