""" PyTorch ChatGLM model. """

import math
import copy
import warnings
import re
import sys

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm
from torch.nn import CrossEntropyLoss, LayerNorm, MSELoss, BCEWithLogitsLoss
from torch.nn.utils import skip_init
from typing import Optional, Tuple, Union, List, Callable, Dict, Any

from transformers import AutoTokenizer

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput


from configuration_chatglm import ChatGLMConfig

# flags required to enable jit fusion kernels

if sys.platform != 'darwin':
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "THUDM/ChatGLM2-6B"
_CONFIG_FOR_DOC = "ChatGLM6BConfig"

CHATGLM_6B_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "THUDM/chatglm2-6b",
    # See all ChatGLM models at https://huggingface.co/models?filter=chatglm
]


def default_init(cls, *args, **kwargs):
    return cls(*args, **kwargs)


class InvalidScoreLogitsProcessor(LogitsProcessor):
    '''
    ## function:
        用于处理模型输出的 logits 分数，将其中的 NaN 和无穷大值替换为 0，并将最后一个维度的第 5 个元素设置为 5e4。相当于我强行让模型去预测到词表中的第五个字符。

    ## Args:
        input_ids：类型为 torch.LongTensor，表示输入的 token ID。
        scores：类型为 torch.FloatTensor，表示模型输出的 logits 分数。 shape = [batch_size, seq_len, vocab_size]
    ## return：
        类型为 torch.FloatTensor，返回处理后的 logits 分数。
        
    __call__ 方法是该类的核心方法，当类的实例被调用时会执行此方法。
    '''
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        '''
        使用 torch.isnan(scores).any() 检查 scores 中是否存在 NaN（Not a Number）值。
            使用 torch.isinf(scores).any() 检查 scores 中是否存在无穷大（Infinity）值。
            如果存在 NaN 或无穷大值，则执行以下操作：
        '''
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_() # 将 scores 中的所有元素置为 0。_ 表示原地操作
            scores[..., 5] = 5e4 # 将 scores 张量的最后一个维度(vocab_size)的第 5 个元素设置为 5e4
            
            # 相当于我强行让模型去预测到词表中的第五个字符。
        return scores


class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config: ChatGLMConfig):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            # 用于计算前缀编码所需的键值对（key-value pairs）的总维度大小
                # config.kv_channels：即，kv_head_size, 表示每个键值对的通道数，即每个键或值向量的维度。
                # kv_size 表示前缀编码所需的键值对的总维度大小。
            kv_size = config.num_layers * config.kv_channels * config.multi_query_group_num * 2
            self.embedding = torch.nn.Embedding(config.pre_seq_len, kv_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(kv_size, config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_size, kv_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len,
                                                config.num_layers * config.kv_channels * config.multi_query_group_num * 2)

    def forward(self, prefix: torch.Tensor):
        '''
        prefix: [batch_size, prefix_length]
        
        output: [batch_size, prefix_length, 2*layers*kv_hidden_size*multi_query_group_num]
        '''
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans.forward(prefix_tokens) # [batch_size, prefix_length, 2*layers*kv_hidden_size*multi_query_group_num]
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values


def split_tensor_along_last_dim(
        tensor: torch.Tensor,
        num_partitions: int,
        contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, original_impl=False, device=None, dtype=None):
        '''
        计算逆频率（inverse frequencies）：1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
            使用 register_buffer 将 inv_freq 注册为模型的非参数张量
            这些频率用于生成不同位置的旋转角度
        '''
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl

    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.


        Args:
            seq_len: sequence length of the input
            n_elem: number of elements in each sequence
            dtype: data type of the input
        
        returns:
            torch.Tensor: [seq_len, n_elem]
        """
        # 计算θ值  (频率)
        # $\Theta = {\theta_i = 10000^{\frac{2(i- 1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

        # 创建位置索引
        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

        # 计算位置索引和θ的外积 
        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float() # shape = [seq_len, n_elem]

        # 计算cos和sin值并堆叠
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)  # shape = [seq_len, n_elem, 2]

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(
            max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
        )


@torch.jit.script   # 这意味着该函数会被编译成 TorchScript，以提高运行效率
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    '''
    此函数的主要作用是将旋转位置编码（Rotary Position Embedding，ROPE）应用到输入张量 x 上。

    rope_cache: 旋转位置编码缓存，用于存储预先计算好的旋转位置编码
        rope_cache.shape = (seq_length, 1, hidden_size) 
    
    x: [sq, b, np, hn]
        x.shape = [sq, b, np, hn]
        sq: 序列长度（sequence length）
        b: 批次大小（batch size）
        np: 注意力头数（number of heads）
        hn: 隐藏层维度（hidden layer dimension）
    函数的执行过程如下：
        首先，将输入张量 x 的最后一个维度（hn）拆分为两个部分：rot_dim 和 x_pass。
        rot_dim 表示需要应用旋转位置编码的维度，而 x_pass 表示不需要应用旋转位置编码的维度。

    '''
    # x: [sq, b, np, hn]
    sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2 # 确定 旋转维度 == 注意力头数 * 2
    
    '''
    x：取输入张量 x 的前 rot_dim 个维度，用于应用旋转位置编码。
    x_pass：取输入张量 x 剩余的维度，这些维度不应用旋转位置编码。
    '''
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2) # shape = [sq, b, 1, np, 2]
    
    # rope_cache[..., 0] 代表 cosθ
    # rope_cache[..., 1] 代表 sinθ
    
    # xshaped[..., 0] 代表 复数 a+bi  的实部 a
    # xshaped[..., 1] 代表 复数 a+bi   的虚部 b
    
    x_out2 = torch.stack(
        [  
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    ) # x_out2.shape = [sq, b, np, hn/2, 2]
    x_out2 = x_out2.flatten(3) #  展平旋转后的维度  shape = [sq, b, np, hn]
    return torch.cat((x_out2, x_pass), dim=-1)


class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        '''
        hidden_states: [batch_size, seq_length, hidden_size]
        
        a_i' = a_i / RMS(a) * g(i), where RMS(a) = sqrt(1/n* sum_{i=1 to n}((a_i)^2))
        
        g(i) 在这个类中，就是可训练参数 self.weight
        '''
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True) # 1/n * sum(x^2)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return (self.weight * hidden_states).to(input_dtype)


class CoreAttention(torch.nn.Module):
    def __init__(self, config: ChatGLMConfig, layer_number):
        super(CoreAttention, self).__init__()

        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        # 完整的 hidden_size
        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head) # 根号 d_k
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff  # 根号 d_k * 层数
        self.coeff = coeff

        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        '''
        query_layer: [seq_length, batch_size, num_head, head_size]
        key_layer: [seq_length, batch_size, num_head, head_size]
        
        torch.nn.functional.scaled_dot_product_attention 已经是 PyTorch 2.0 提供的高效实现，一般不需要额外的性能优化。
        '''
        pytorch_major_version = int(torch.__version__.split('.')[0])
        if pytorch_major_version >= 2:
            # 将 query_layer, key_layer, value_layer 的维度进行重排
            query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]
            if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
                # 使用 scaled_dot_product_attention 函数计算上下文层，开启因果掩码
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                 is_causal=True)
            else:
                if attention_mask is not None:
                    # 如果提供了注意力掩码，将其取反
                    attention_mask = ~attention_mask
                context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                                    attention_mask)
            # 将上下文层的维度进行重排，恢复到原始的维度顺序
            #  context_layer 在调用 permute 之前的形状为 [batch_size, num_head, seq_length, head_size]。
            # 重排后形状: [seq_length, batch_size, num_head, head_size]                                                                     
            context_layer = context_layer.permute(2, 0, 1, 3)
            # 取 context_layer 形状的前两个维度，然后将 self.hidden_size_per_partition 作为最后一个维度添加进去。
            # new_context_layer_shape 为 (seq_length, batch_size, self.hidden_size_per_partition)。
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.reshape(*new_context_layer_shape)
        else:
            # Raw attention scores
            
            # query_layer: [seq_length, batch_size, num_head, head_size]
            # key_layer: [seq_length, batch_size, num_head, head_size]

            # [b, np, sq, sk] = [batch_size, num_partitions, seq_len_query, seq_len_key]  注：num_partitions == num_attention_heads
            output_size = (query_layer.size(1), query_layer.size(2), query_layer.size(0), key_layer.size(0))

            # [sq, b, np, hn] -> [sq, b * np, hn] = [seq_len_query, batch_size * num_partitions, head_size]
            query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
            # [sk, b, np, hn] -> [sk, b * np, hn]
            key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

            # preallocting input tensor: [b * np, sq, sk]
            matmul_input_buffer = torch.empty(
                output_size[0] * output_size[1], output_size[2], output_size[3], dtype=query_layer.dtype,
                device=query_layer.device
            )

            # Raw attention scores. [b * np, sq, sk]
            matmul_result = torch.baddbmm(
                matmul_input_buffer,
                query_layer.transpose(0, 1),  # [b * np, sq, hn]
                key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / self.norm_factor),
            )

            # change view to [b, np, sq, sk]
            attention_scores = matmul_result.view(*output_size)

            # ===========================
            # Attention probs and dropout
            # ===========================

            # attention scores and attention mask [b, np, sq, sk]
            if self.attention_softmax_in_fp32:
                attention_scores = attention_scores.float()
            if self.coeff is not None:
                attention_scores = attention_scores * self.coeff
            if attention_mask is None and attention_scores.shape[2] == attention_scores.shape[3]:
                attention_mask = torch.ones(output_size[0], 1, output_size[2], output_size[3],
                                            device=attention_scores.device, dtype=torch.bool) # shape = [batch_size, 1, seq_len_query, seq_len_key]
                attention_mask.tril_() # 下三角全1
                attention_mask = ~attention_mask #下三角全0， 上三角全1
            if attention_mask is not None:
                # 矩阵中为True的地方填充给定值 "-inf"
                attention_scores = attention_scores.masked_fill(attention_mask, float("-inf"))
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = attention_probs.type_as(value_layer)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.attention_dropout(attention_probs)
            # =========================
            # Context layer. [sq, b, hp]   , hp = num_heads * head_size
            # =========================

            # value_layer -> context layer.
            # [sk, b, np, hn] --> [b, np, sq, hn]
            
            # query_layer.shape = [sq, b, np, hn]    [sq, b * np, hn]
            # value_layer.shape = [sk, b, np, hn]    [sk, b * np, hn]

            # context layer shape: [b, np, sq, hn]
            output_size = (value_layer.size(1), value_layer.size(2), query_layer.size(0), value_layer.size(3))
            # change view [sk, b * np, hn]
            value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
            # change view [b * np, sq, sk]
            attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
            # matmul: [b * np, sq, hn]
            context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
            # change view [b, np, sq, hn]
            context_layer = context_layer.view(*output_size)
            # [b, np, sq, hn] --> [sq, b, np, hn]
            context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
            # [sq, b, np, hn] --> [sq, b, hp]
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class SelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h] = [seq_length, batch_size, hidden_size]
    and returns output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(SelfAttention, self).__init__()
        self.layer_number = max(1, layer_number)

        # model's hidden_size,  if model == bert then hidden_size = 768
        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = ( # 包含所有 qkv 注意力头的隐单元维度， 假设一共可以分成8个头，那么query占8个头，kv分别只占了1个头 （如果group_num == 1)
                    # query.hidden_size  + (key.hidden_size_per_head + value.hidden_size_per_head) * num_multi_query_groups
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        self.query_key_value = nn.Linear(config.hidden_size, self.qkv_hidden_size,
                                         bias=config.add_bias_linear or config.add_qkv_bias,
                                         device=device, **_config_to_kwargs(config)
                                         )

        self.core_attention = CoreAttention(config, self.layer_number)

        # Output.
        self.dense = nn.Linear(self.projection_size, config.hidden_size, bias=config.add_bias_linear,
                               device=device, **_config_to_kwargs(config)
                               )

    def _allocate_memory(self, inference_max_sequence_len, batch_size, device=None, dtype=None):
        if self.multi_query_attention:
            num_attention_heads = self.num_multi_query_groups_per_partition
        else:
            num_attention_heads = self.num_attention_heads_per_partition
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            num_attention_heads,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
    ):
        '''
        rotary_pos_emb: shape = (seq_length, 1, hidden_size)
        '''
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]   
        # 准确来说是 [seq_len, batch_size, num_partitions * (query_hidden_size + key_hidden_size + value_hidden_size)]
        mixed_x_layer = self.query_key_value(hidden_states)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            '''
            query_layer.shape = [seq_len, batch_size, num_attention_heads_per_partition * hidden_size_per_attention_head]
            key_layer.shape = [seq_len, batch_size, num_multi_query_groups_per_partition * hidden_size_per_attention_head]
            value_layer.shape = [seq_len, batch_size, num_multi_query_groups_per_partition * hidden_size_per_attention_head]
            '''
            query_layer = query_layer.view(
                query_layer.size()[:-1] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1] + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (self.num_multi_query_groups_per_partition, self.hidden_size_per_attention_head)
            )
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                               (self.num_attention_heads_per_partition,
                                3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)

        # apply relative positional encoding (rotary embedding)
        if rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        # adjust key and value for inference
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            key_layer = torch.cat((cache_k, key_layer), dim=0)
            value_layer = torch.cat((cache_v, value_layer), dim=0)
        if use_cache:
            kv_cache = (key_layer, value_layer)
        else:
            kv_cache = None

        if self.multi_query_attention: # 复制key和value, 使得 key-value pair的数量 == query的数量
            key_layer = key_layer.unsqueeze(-2) # [seq_len, batch_size, num_multi_query_groups_per_partition, 1, hidden_size_per_attention_head]
            
            key_layer = key_layer.expand(  # 这样做的目的是为了让每个查询头都能对应一个键头。
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            ) 
            
            # self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition 表示每个组内有多少query头
            
            # 由于， num_multi_query_groups * num_heads // num_multi_query_groups = num_heads
            key_layer = key_layer.contiguous().view( # 将扩展后的 key_layer 重新调整形状，使其与 query_layer 的形状一致
                key_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            ) # shape = [seq_len, batch_size, num_attention_heads_per_partition, hidden_size_per_attention_head]
            
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.expand(
                -1, -1, -1, self.num_attention_heads_per_partition // self.num_multi_query_groups_per_partition, -1
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:2] + (self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
            )

        # ==================================
        # core attention computation
        # ==================================

        context_layer = self.core_attention.forward(query_layer, key_layer, value_layer, attention_mask)

        # =================
        # Output. [sq, b, h] = [seq_length, batch_size, hidden_size]
        # =================

        output = self.dense(context_layer)

        return output, kv_cache


def _config_to_kwargs(args):
    common_kwargs = {
        "dtype": args.torch_dtype,
    }
    return common_kwargs


class MLP(torch.nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config: ChatGLMConfig, device=None):
        super(MLP, self).__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=self.add_bias,
            device=device,
            **_config_to_kwargs(config)
        )

    def forward(self, hidden_states):
        # [s, b, 4hp] = [seq_length, batch_size, 4*hidden_size]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(GLMBlock, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                             dtype=config.torch_dtype)

        # Self attention.
        self.self_attention = SelfAttention(config, layer_number, device=device)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                                      dtype=config.torch_dtype)

        # MLP
        self.mlp = MLP(config, device=device)

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
    ):
        '''
        rotary_pos_emb: shape = (seq_length, 1, hidden_size)
        
        hidden_states: [s, b, h]
        '''
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, kv_cache = self.self_attention.forward(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output

        return output, kv_cache


class GLMTransformer(torch.nn.Module):
    """Transformer class."""

    def __init__(self, config: ChatGLMConfig, device=None):
        super(GLMTransformer, self).__init__()

        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        # Transformer layers.
        def build_layer(layer_number):
            return GLMBlock(config, layer_number, device=device)

        self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                                 dtype=config.torch_dtype)

        self.gradient_checkpointing = False

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_caches=None,
            use_cache: Optional[bool] = True,
            output_hidden_states: Optional[bool] = False,
    ):
        '''
        hidden_states: aka. last_hidden_states, shape= (batch_size, seq_length, hidden_size)

        kv_caches: aka., past_key_values, shape= (num_layers, 2, batch_size, num_heads, seq_length, head_dim)

        rotary_pos_emb: shape = (seq_length, 1, hidden_size)
        '''
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        presents = () if use_cache else None # 每一层的 kv-cache 的集合
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_self_attentions = None
        all_hidden_states = () if output_hidden_states else None
        for index in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer:GLMBlock = self._get_layer(index)
            
            '''
            checkpoint: 传播所需的张量一直保持到它们在梯度计算中被使用，而是在检查点区域的正向计算中省略为反向传播保存张量，
                并在反向传播过程中重新计算它们。激活检查点可以应用于模型的任何部分。
            '''
            if self.gradient_checkpointing and self.training:
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches[index],
                    use_cache
                )
            else:
                layer_ret = layer.forward(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache
                )
            hidden_states, kv_cache = layer_ret
            if use_cache:
                presents = presents + (kv_cache,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, presents, all_hidden_states, all_self_attentions


class ChatGLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """
    is_parallelizable = False
    supports_gradient_checkpointing = True
    config_class = ChatGLMConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["GLMBlock"]

    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        return

    def get_masks(self, input_ids, past_key_values, padding_mask=None):
        '''
        根据输入的 input_ids、past_key_values 和 padding_mask 生成注意力掩码。

        参数:
        input_ids (torch.Tensor): 输入的 token ID 张量，形状为 [batch_size, seq_length]。

        past_key_values (List[Tuple[Tensor]]): 
            past_key_values 是一个存储过去的键（key）和值（value）张量的元组列表，
            通常用于在自回归生成过程中缓存之前步骤的键值对，以避免重复计算。

            past_key_values 是一个长度为 num_layers 的列表，列表中的每个元素是一个元组 (key, value)。
            key 和 value 的形状均为 (past_length, batch_size, num_heads, head_dim)。

        padding_mask (torch.Tensor or None): 填充掩码张量，如果没有则为 None。
            shape = (batch_size, seq_length)

        返回:
        torch.Tensor: 生成的注意力掩码张量，形状为 [batch_size, 1, seq_length, seq_length + past_length]。
        
        '''
        batch_size, seq_length = input_ids.shape
        #  创建一个全为 1 的三维张量作为初始的全注意力掩码
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_ids.device) # [batch_size, seq_length, seq_length]  
        # 将全注意力掩码转换为下三角矩阵，确保每个位置只能关注到其之前的位置
        full_attention_mask.tril_()
        # 初始化过去序列的长度为 0
        past_length = 0
        if past_key_values:
            # past_key_values[0][0] 表示第一层的键张量，
            # 其形状为 (past_length, batch_size, num_heads, head_dim)
            past_length = past_key_values[0][0].shape[0]
        if past_length:
            # 创建一个全为 1 的三维张量，表示过去序列的注意力掩码
            past_attention_mask = torch.ones(batch_size, seq_length, past_length, device=input_ids.device)
            # 将过去序列的注意力掩码和当前序列的注意力掩码在最后一个维度上拼接
            full_attention_mask = torch.cat((past_attention_mask, full_attention_mask), dim=-1)
            
            # 拼接后， shape = (batch_size, seq_length, seq_length + past_length)
            
        if padding_mask is not None:
            # 将填充掩码扩展一个维度，并与全注意力掩码相乘，以应用填充掩码
            # padding_mask 只管当前的序列看不看得到，不会管历史的
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            # 处理没有历史信息但有填充掩码的情况
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool() # True值表示该位置的注意力权重会被设为0（即被屏蔽）, False值表示该位置可以参与注意力计算
        full_attention_mask.unsqueeze_(1) # shape = (batch_size, 1, seq_length, seq_length + past_length)
        return full_attention_mask
    
        '''
        假设 batch_size == 1, 
        padding_mask = [0, 0, 1]
        padding_mask.unsqueeze(1) 的结果
        
        0 0 1
        0 0 1
        1 1 1
        '''

    def get_position_ids(self, input_ids, device):
        batch_size, seq_length = input_ids.shape
        # 使用 repeat 方法将二维张量在第 0 维上重复 batch_size 次，得到形状为 (batch_size, seq_length) 的二维张量
        # 这样，每个批次中的序列都有相同的位置编码
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        return position_ids

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GLMTransformer):
            module.gradient_checkpointing = value


class Embedding(torch.nn.Module):
    """Language model embeddings."""

    def __init__(self, config: ChatGLMConfig, device=None):
        super(Embedding, self).__init__()

        self.hidden_size = config.hidden_size
        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size,
            self.hidden_size,
            dtype=config.torch_dtype,
            device=device
        )
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids):
        '''
        return embeddings, shape = [seq_len, batch_size, hidden_size]
        '''
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()
        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


class ChatGLMModel(ChatGLMPreTrainedModel):
    def __init__(self, config: ChatGLMConfig, device=None, empty_init=True):
        super().__init__(config)
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        init_kwargs = {}
        if device is not None:
            init_kwargs["device"] = device
        self.embedding = init_method(Embedding, config, **init_kwargs)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels  # key, value 的 head_size

        # Rotary positional embeddings
        self.seq_length = config.seq_length
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )
        # why rotary_dim // 2 ? 因为 我们把 样本 x 的 hidden_size 分成了偶数和奇数维， 取的时候实际: x_{2k}, x_{2k+b}, 分别作为复数向量的实部和虚部
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, original_impl=config.original_rope, device=device,
                                              dtype=config.torch_dtype)
        self.encoder = init_method(GLMTransformer, config, **init_kwargs)
        self.output_layer = init_method(nn.Linear, config.hidden_size, config.padded_vocab_size, bias=False,
                                        dtype=config.torch_dtype, **init_kwargs)
        # 前缀序列长度， 用于 p-tuning v2
        self.pre_seq_len = config.pre_seq_len
        self.prefix_projection = config.prefix_projection
        if self.pre_seq_len is not None:
            for param in self.parameters(): # 冻结 base model 的参数
                param.requires_grad = False 
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            self.prefix_encoder = PrefixEncoder(config)
            self.dropout = torch.nn.Dropout(0.1)

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def get_prompt(self, batch_size, device, dtype=torch.half):
        '''
        用于 P-Tuning v2 中生成连续提示向量
        将前缀tokens转换为key-value缓存格式
        返回经过处理的past_key_values用于注意力计算
        '''
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device) # shape = (batch_size, pre_seq_len)
        past_key_values:torch.Tensor = self.prefix_encoder.forward(prefix_tokens).type(dtype) # shape = [batch_size, prefix_length, 2*layers*kv_hidden_size*multi_query_group_num]
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.num_layers * 2,
            self.multi_query_group_num,  # key-value pairs number
            self.kv_channels    # kv_head_size
        )
        # seq_len, b, nh, hidden_size
        past_key_values = self.dropout(past_key_values)
        # split(2) 会将这个张量沿着第 0 维分割成多个大小为 2 的子张量，最终返回一个包含这些子张量的元组。
        past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
        return past_key_values

    def forward(
            self,
            input_ids,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):  
        '''
        input_ids: shape = (batch_size, seq_length)
        output_hidden_states: 是否返回所有层的隐层状态, shape = (num_layers, batch_size, seq_length, hidden_size)
        attention_mask: 等同于 `padding_mask`, shape = (batch_size, seq_length)
        full_attention_mask: shape = (batch_size, seq_length, seq_length + past_length)
        past_key_values: shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        inputs_embeds: shape = (batch_size, seq_length, hidden_size)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,  shape = (seq_length, batch_size, hidden_size)
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        '''
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if self.pre_seq_len is not None: # 使用前缀微调
            if past_key_values is None: # 如果 past_key_values 为空，说明还没有生成前缀的键值对。调用 self.get_prompt 方法生成前缀的键值对。
                past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device,
                                                  dtype=inputs_embeds.dtype)
            if attention_mask is not None: # 存在 padding_mask
                # 如果存在padding掩码 attention_mask，则需要将前缀序列的padding掩码（全1）添加到原有的padding掩码中。
                attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)),
                                            attention_mask], dim=-1) # shape = (batch_size, seq_length + pre_seq_len)

        if full_attention_mask is None:
            # padding 掩码存在且非全1， 则需要综合历史序列，当前序列和padding，来生成完整的掩码
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length) # shape = (seq_length, hidden_size)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            # rotary_pos_emb[None, :seq_length] 会选择 rotary_pos_emb 中前 seq_length 个位置的嵌入，并在第0维添加一个维度，形状变为 [1, seq_length, ...]。
            rotary_pos_emb = rotary_pos_emb[None, :seq_length] # shape = (1, seq_length, hidden_size)
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous() # shape = (seq_length, 1, hidden_size)

        # Run encoder.
        # presents： 如果 use_cache=True，则返回一个元组，其中包含每个编码器层的键值对缓存。
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder.forward(
            inputs_embeds, full_attention_mask, rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values, use_cache=use_cache, output_hidden_states=output_hidden_states
        )

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def quantize(self, weight_bit_width: int):
        from .quantization import quantize
        quantize(self.encoder, weight_bit_width)
        return self


class ChatGLMForConditionalGeneration(ChatGLMPreTrainedModel):
    def __init__(self, config: ChatGLMConfig, empty_init=True, device=None):
        super().__init__(config)

        self.max_sequence_length = config.max_length
        self.transformer = ChatGLMModel(config, empty_init=empty_init, device=device)
        self.config = config
        self.quantized = False

        if self.config.quantization_bit:
            self.quantize(self.config.quantization_bit, empty_init=True)

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, # standardize_cache_format=standardize_cache_format
        )

        # update attention mask
        # attention_mask.shape = (batch_size, seq_length)
        # 如果 model_kwargs 中包含 attention_mask，则在其末尾添加一个全为 1 的列，以适应新生成的 token。
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        # update position ids
        # position_ids.shape = (batch_size, seq_length)
        # 如果 model_kwargs 中包含 position_ids，则复制最后一个位置 ID 并加 1，然后将其添加到 position_ids 的末尾。
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()  # shape = (batch_size, 1)
            new_position_id += 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1
            )

        model_kwargs["is_first_forward"] = False  # 表示不是第一次前向传播
        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            is_first_forward: bool = True,
            **kwargs
    ) -> dict:
        # only last token for input_ids if past is not None
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids, device=input_ids.device)
        if not is_first_forward:
            '''
            在生成过程中，当past_key_values存在时，模型已经处理过前面的序列，
                并且缓存了对应的key和value。此时，模型只需要处理最新的token，
                所以仅需获取这个最新token的位置信息。
                
            此时，attention中的 query 只包含最后一个token的嵌入，不会包含 past_key_values
            '''
            if past_key_values is not None: # 只取 position_ids 和 input_ids 的最后一个元素。
                position_ids = position_ids[..., -1:]
                input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "return_last_logit": True,
            "use_cache": use_cache
        }

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = False,
    ):
        '''
        input_ids.shape = (batch_size, seq_length)
        labels.shape = (batch_size, seq_length)
        '''
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0] #shape = [seq_length, batch_size, hidden_size]
        if return_last_logit: # 只取最后一个时间步的输出
            hidden_states = hidden_states[-1:]  # 只取最后一个输出 shape = [1, batch_size, hidden_size]
        lm_logits = self.transformer.output_layer(hidden_states) # shape = [1, batch_size, vocab_size]
        lm_logits = lm_logits.transpose(0, 1).contiguous() # shape = [batch_size, 1, vocab_size]

        loss = None
        
        # 如果提供了标签，则计算损失
        if labels is not None:
            # 将 lm_logits 转换为 torch.float32 类型，因为交叉熵损失函数通常要求输入为 float32 类型
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            # 使得第 n 个 token 的预测目标是第 n+1 个 token
            # shift_logits 的形状为 [batch_size, seq_len-1, vocab_size]
                # 假设，完整的句子是 ：“今天天气很不错”
            shift_logits = lm_logits[..., :-1, :].contiguous() # 今天天气很不 shape = [batch_size, seq_len-1, vocab_size]
            shift_labels = labels[..., 1:].contiguous() # 天天气很不错  shape = [batch_size, seq_len-1]
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            
            # shape =  (loss, lm_logits, transformer_outputs[1], transformer_outputs[2], ...)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
            past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        
        ---
        
        该函数用于在beam search生成过程中重新排序缓存中的键值对，
            这样做的目的是确保在每一步生成过程中，past_key_values 能够与正确的束索引（beam index）相对应，
            从而保证生成的候选序列与历史缓存正确对齐。
        
        通过index_select方法按beam_idx索引重新选择各层的key和value张量，
            保持内存共享以提高效率。
        ---
        
        Args:
        past：这是一个嵌套元组，外层元组的每个元素代表模型的一层，内层元组包含两个 torch.Tensor，分别是该层的 key 和 value 张量。
            通常 key 和 value 的形状为 (batch_size, num_heads, seq_length, head_dim)。
        
        beam_idx：这是一个 torch.LongTensor 类型的张量，包含了束搜索过程中每个束的索引，用于指示如何对 past_key_values 进行重新排序。
        
        Return:
        返回值：返回一个与 past 结构相同的嵌套元组，其中的 key 和 value 张量已经按照 beam_idx 进行了重新排序。
        """
        
        '''
        index_select(1, beam_idx.to(layer_past[0].device))：
            使用 index_select 方法在第 1 个维度（通常是 batch_size 维度）上按照 beam_idx 进行索引选择。
            beam_idx.to(layer_past[0].device) 确保 beam_idx 与 key 或 value 张量在同一设备上。
        '''
        return tuple(
            (
                layer_past[0].index_select(1, beam_idx.to(layer_past[0].device)), # key
                layer_past[1].index_select(1, beam_idx.to(layer_past[1].device)), # value
            )
            for layer_past in past
        )

    def process_response(self, response):
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        return response

    def build_inputs(self, tokenizer:AutoTokenizer, query: str, history: List[Tuple[str, str]] = None):
        prompt = tokenizer.build_prompt(query, history=history)
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs

    def build_stream_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        '''
        ## function:
            其主要功能是根据用户输入的查询文本 query 和对话历史记录 history 构建模型输入
        '''
        if history:
            prompt = "\n\n[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_ids = input_ids[1:] # 去掉 input_ids 的第一个元素，可能是为了去除不必要的起始 token。
            # batch_encode_plus : 将多个文本批量编码为模型可以处理的输入格式
            inputs = tokenizer.batch_encode_plus([(input_ids, None)], return_tensors="pt", add_special_tokens=False)
        else:
            prompt = "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
            inputs = tokenizer([prompt], return_tensors="pt")
            
        '''
        inputs 的类型是一个字典（dict），这是 transformers 库中分词器的常见返回类型。字典中通常包含以下几个键：

            input_ids：这是一个 torch.Tensor 类型的张量，包含输入文本的 token ID。
            attention_mask：这也是一个 torch.Tensor 类型的张量，用于指示哪些位置是有效的输入，哪些位置是填充的。
        '''
        inputs = inputs.to(self.device)
        return inputs

    @torch.inference_mode() # 这是一个装饰器，用于开启推理模式，在该模式下，PyTorch 会禁用梯度计算，从而减少内存使用并提高推理速度。
    def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 8192, num_beams=1,
             do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None, **kwargs):
        '''
        ## Args:
            tokenizer：分词器，用于将文本转换为模型可以处理的 token。
            query：用户输入的查询文本，类型为字符串。
            history：对话历史记录，类型为元组列表，每个元组包含两个字符串，分别表示用户的输入和模型的回复。默认为 None。
            max_length：生成文本的最大长度，默认为 8192。
            num_beams：束搜索的束数，默认为 1。
            do_sample：是否进行采样，默认为 True。
            top_p：采样时的概率阈值，默认为 0.8。
            temperature：采样时的温度参数，默认为 0.8。
            logits_processor：对数概率处理器，默认为 None。
            
        ## Return
            return response, history
        
        ## 功能：
        该方法的主要功能是与模型进行对话交互，具体步骤如下：

            1. 初始化对话历史记录和对数概率处理器。
            2. 设置生成文本所需的参数。
            3. 根据分词器、查询文本和对话历史记录构建模型的输入。
            4. 调用 generate 方法生成文本。
            5. 解码生成的输出，并对回复进行处理。
            6. 更新对话历史记录。
            7. 返回处理后的回复和更新后的对话历史记录。
        
        '''
        
        
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor()) # 用于处理无效的对数概率
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        inputs = self.build_inputs(tokenizer, query, history=history)
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):] # 将生成的输出转换为列表，并截取输入部分之后的内容。
        response = tokenizer.decode(outputs)
        response = self.process_response(response)
        history = history + [(query, response)]
        return response, history

    @torch.inference_mode()
    def stream_chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, past_key_values=None,
                    max_length: int = 8192, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None,
                    return_past_key_values=False, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        
        gen_kwargs = {"max_length": max_length, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        if past_key_values is None and not return_past_key_values:
            inputs = self.build_inputs(tokenizer, query, history=history)
        else:
            inputs = self.build_stream_inputs(tokenizer, query, history=history)
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[0] # 这里取第一个键张量的第一个维度（即序列长度）。
            if self.transformer.pre_seq_len is not None:
                past_length -= self.transformer.pre_seq_len # pre_seq_len 可能表示前缀序列的长度，这里是为了排除前缀序列的影响。
            inputs.position_ids += past_length # 在原有的 位置id矩阵的基础上， 加上历史上生成的文本的长度， 表示当前生成的文本的起始位置是从历史上生成的文本的末尾开始的。
            attention_mask = inputs.attention_mask
            attention_mask = torch.cat((attention_mask.new_ones(1, past_length), attention_mask), dim=1)
            inputs['attention_mask'] = attention_mask
            
        for outputs in self.stream_generate(**inputs, past_key_values=past_key_values,
                                            return_past_key_values=return_past_key_values, **gen_kwargs):
            if return_past_key_values:
                outputs, past_key_values = outputs
            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]  # shape = (1, num_generated_tokens)
            response = tokenizer.decode(outputs)
            
            if response and response[-1] != "�":
                response = self.process_response(response)
                new_history = history + [(query, response)]
                if return_past_key_values:
                    yield response, new_history, past_key_values
                else:
                    yield response, new_history

    @torch.inference_mode()
    def stream_generate(
            self,
            input_ids,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            return_past_key_values=False,
            **kwargs,
    ):
        batch_size, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]

        if generation_config is None:
            generation_config = self.generation_config
            
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        model_kwargs["use_cache"] = generation_config.use_cache
        bos_token_id, eos_token_id = generation_config.bos_token_id, generation_config.eos_token_id

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            warnings.warn(
                f"Using `max_length`'s default ({generation_config.max_length}) to control the generation length. "
                "This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we"
                " recommend using `max_new_tokens` to control the maximum length of the generation.",
                UserWarning,
            ) # 这里的作用是将 generation_config.max_length设置为generation_config.max_new_tokens + input_ids_seq_length
        elif generation_config.max_new_tokens is not None:
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
            if not has_default_max_length:
                logger.warn(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)",
                    UserWarning,
                )

        if input_ids_seq_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_seq_length,
            encoder_input_ids=input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
        )

        stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        logits_warper = self._get_logits_warper(generation_config)

        # 用于标记每个序列是否完成生成
        # input_ids.new(...)：借助 input_ids 张量的 new 方法创建一个新的张量，新张量会和 input_ids 处于相同的设备（如 CPU 或 GPU）上。
        # .fill_(1)：把新张量的所有元素初始化为 1。在这个上下文中，值为 1 表示对应序列还未完成生成，值为 0 则表示对应序列已经完成生成。
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1) # shape = (batch_size, )
        scores = None
        while True:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            outputs = self.forward(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            if generation_config.do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1) # 多项式采样， 随机选择一个 token
            else:
                next_tokens = torch.argmax(probs, dim=-1)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)  # shape = (batch_size, seq_length + 1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            
            # mul 是 torch.Tensor 的乘法方法，用于将 unfinished_sequences 与上述求和结果相乘
            # next_tokens != i： 整个tensor中都没有停止符，才返回一个1， 否则返回0。 
            unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in eos_token_id)).long())
            
            '''
            假设 eos_token_id = [100]，next_tokens = [101, 100, 102]，unfinished_sequences = [1, 1, 1]。

            next_tokens != 100 的结果为 [True, False, True]，转换为数值为 [1, 0, 1]。
            unfinished_sequences.mul([1, 0, 1]) 的结果为 [1, 0, 1]，表示第二个序列已经完成生成。
            '''
            if return_past_key_values:
                yield input_ids, outputs.past_key_values
            else:
                yield input_ids
            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

    def quantize(self, bits: int, empty_init=False, device=None, **kwargs):
        '''
        quantization 包已经过时了， 我们直接忽略这个函数
        '''
        if bits == 0:
            return

        from .quantization import quantize  

        if self.quantized:
            logger.info("Already quantized.")
            return self

        self.quantized = True

        self.config.quantization_bit = bits

        self.transformer.encoder = quantize(self.transformer.encoder, bits, empty_init=empty_init, device=device,
                                            **kwargs)
        return self


class ChatGLMForSequenceClassification(ChatGLMPreTrainedModel):
    def __init__(self, config: ChatGLMConfig, empty_init=True, device=None):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.transformer = ChatGLMModel(config, empty_init=empty_init, device=device)

        self.classifier_head = nn.Linear(config.hidden_size, config.num_labels, bias=True, dtype=torch.half)
        if config.classifier_dropout is not None:
            self.dropout = nn.Dropout(config.classifier_dropout)
        else:
            self.dropout = None
        self.config = config

        if self.config.quantization_bit:
            self.quantize(self.config.quantization_bit, empty_init=True)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            full_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, ...], SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            full_attention_mask=full_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0] # shape = (seq_length, batch_size, hidden_size)
        pooled_hidden_states = hidden_states[-1] # shape = (batch_size, hidden_size)
        if self.dropout is not None:
            pooled_hidden_states = self.dropout(pooled_hidden_states) 
        logits = self.classifier_head(pooled_hidden_states) # shape = (batch_size, num_labels)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze().float(), labels.squeeze()) # (batch_size, ) * (batch_size,)
                else:
                    loss = loss_fct(logits.float(), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels).float(), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits.float(), labels.view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
