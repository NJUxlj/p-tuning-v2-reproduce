# coding=utf-8
# Copyright 2020 Microsoft and the Hugging Face Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch DeBERTa model. """

import math
from collections.abc import Sequence

import torch
from torch import _softmax_backward_data, nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.deberta.configuration_deberta import DebertaConfig


from typing import List, Dict, Union, Tuple, Optional, Callable, Any

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DebertaConfig"
_TOKENIZER_FOR_DOC = "DebertaTokenizer"
_CHECKPOINT_FOR_DOC = "microsoft/deberta-base"

DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/deberta-base",
    "microsoft/deberta-large",
    "microsoft/deberta-xlarge",
    "microsoft/deberta-base-mnli",
    "microsoft/deberta-large-mnli",
    "microsoft/deberta-xlarge-mnli",
]


class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config
 

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size
    
    





class XSoftmax(torch.autograd.Function):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (:obj:`torch.tensor`): The input tensor that will apply softmax.
        mask (:obj:`torch.IntTensor`): The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax

    Example::

          >>> import torch
          >>> from transformers.models.deberta.modeling_deberta import XSoftmax

          >>> # Make a tensor
          >>> x = torch.randn([4,20,100])

          >>> # Create a mask
          >>> mask = (x>0).int()

          >>> y = XSoftmax.apply(x, mask, dim=-1)
    """

    @staticmethod
    def forward(self, input, mask, dim):
        self.dim = dim
        rmask = ~(mask.bool())

        output = input.masked_fill(rmask, float("-inf"))
        output = torch.softmax(output, self.dim)
        output.masked_fill_(rmask, 0)
        # 保存输出结果，用于反向传播
        '''
        该方法用于在 setup_context 或 forward 方法中保存需要在反向传播时使用的张量。
        这些张量会被存储在 self.to_save 中，在反向传播时可以通过 ctx.saved_tensors 属性访问。
        '''
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        '''
        通过实现 backward 方法，支持自动求导，使得该类可以在 PyTorch 的自动求导系统中正常使用。
        '''
        (output,) = self.saved_tensors
        # 调用 PyTorch 的 _softmax_backward_data 函数计算输入的梯度
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output)
        return inputGrad, None, None


class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None






class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack: List = None

    def forward(self, x):
        """
        Call the module

        Args:
            x (:obj:`torch.tensor`): The input tensor to apply dropout
        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob
        
        
        
        
### 主入口开始

class DebertaLayerNorm(nn.Module):
    """LayerNorm module in the TF style (epsilon inside the square root)."""

    def __init__(self, size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_type = hidden_states.dtype
        hidden_states = hidden_states.float() 
        mean = hidden_states.mean(-1, keepdim=True) # shape= [batch_size, seq_len, 1] 
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.to(input_type)
        # weight 广播到 hidden_states.shape
        y = self.weight * hidden_states + self.bias
        return y


class DebertaSelfOutput(nn.Module):
    '''
    相当于 FFN
    '''
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
    
    
    

class DebertaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.output = DebertaSelfOutput(config)
        self.config = config

    def forward(
        self,
        hidden_states,
        attention_mask,
        return_att=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        past_key_value=None,
    ):
        self_output = self.self.forward(
            hidden_states,
            attention_mask,
            return_att,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            past_key_value=past_key_value,
        )
        if return_att:
            self_output, att_matrix = self_output
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states)

        if return_att:
            return (attention_output, att_matrix)
        else:
            return attention_output
        



# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Deberta
class DebertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense  = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
            
    def forward(self, hidden_states):
        hidden_states = self.dense.forward(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
        
        

class DebertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
        
        

class DebertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = DebertaAttention(config)
        self.intermediate = DebertaIntermediate(config)
        self.output = DebertaOutput(config)
        
        
    def forward(
        self,
        hidden_states,
        attention_mask,
        return_att = False,
        query_states = None,
        relative_pos = None,
        rel_embeddings = None,
        past_key_value = None,
    ):
        '''
        ##Args:
            :param: hidden_states, shape = [bz, seq_len, d]
            :param: attention_mask, shape = [bz, seq_len]
            :param: return_att, bool
            :param: query_states, shape = [bz, seq_len, d]
            :param: relative_pos, shape = [1, q_size, k_size], 功能：计算相对位置编码
            :param: rel_embeddings, shape = [2k, d]， 功能：计算相对位置编码
            :param: past_key_value, shape = [n_layers, bz, seq_len, d]
            
            
        ##Fucntion:
            rel_embeddings 通常是相对位置嵌入矩阵 R，
                        它包含了相对位置之间的关系信息，用于增强模型对相对位置的建模能力。

                    
                以下是标准注意力+相对位置嵌入：

                \[
                \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + QR^T}{\sqrt{d_k}}\right)V
                \]
            
            
            
            1. relative_pos（相对位置索引）
            relative_pos 是一个张量，其形状通常为 [1, query_size, key_size]。
                它表示查询序列中每个位置与键序列中每个位置之间的相对位置。具体来说，对于查询序列中的每个位置 q 和键序列中的每个位置 k，relative_pos 存储了 q - k 的值。这个相对位置信息有助于模型捕捉序列中元素之间的相对位置关系。

            2. relative_embedding（相对位置嵌入）
            relative_embedding 是一个张量，其形状通常为 [2 * max_relative_positions, hidden_size]。
                它是一个预定义的嵌入矩阵，用于将相对位置索引 relative_pos 映射到一个高维向量空间中。这个嵌入矩阵包含了相对位置之间的关系信息，用于增强模型对相对位置的建模能力。

            两者之间的关系
            relative_pos 是一个整数索引矩阵，用于从 relative_embedding 中查找对应的相对位置嵌入向量。具体来说，对于 relative_pos 中的每个元素 i，我们可以通过 relative_embedding[i + max_relative_positions] 来获取对应的相对位置嵌入向量。

            换言之：
                relative_pos用于提供相对位置， relative_embedding提供了相对位置的嵌入表示
                    
        '''
        attention_output = self.attention.forward(
            hidden_states,
            attention_mask,
            return_att = return_att,
            query_states = query_states,
            relative_pos = relative_pos,
            rel_embeddings = rel_embeddings,
            past_key_value = past_key_value,
        )
        
        if return_att:
            attention_output, att_matrix = attention_output
        
        intermediate_output  = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        if return_att:
            return (layer_output, att_matrix)
        else:
            return layer_output
        
            
        



class DebertaEncoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([DebertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.relative_attention = getattr(config, "relative_attention", False)
        if self.relative_attention:
            # 论文中的 k
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.rel_embeddings = nn.Embedding(self.max_relative_positions * 2, config.hidden_size) # shape = (2k, d)

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        return rel_embeddings
    

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2: # shape = (batch_size, seq_len) -> shape = [bz, 1, seq_len] -> shape = [bz, 1, seq_len, seq_len]
           extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
           attention_mask = attention_mask.byte()
           
       
        elif attention_mask.dim() == 3: # shape = [bz, seq_len, seq_len]
           attention_mask = attention_mask.unsqueeze(1) # shape = [bz, 1, seq_len, seq_len]
        
        
        return attention_mask
    
    
    def get_rel_pos(self, hidden_states, query_states = None, relative_pos = None):
        '''
        return :relative_pos:`torch.LongTensor`: A tensor with shape [1, query_size, key_size]
        '''
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(q, hidden_states.size(-2), device=hidden_states.device)
        
        return relative_pos
    
    
    
    def forward(
        self,
        hidden_states, 
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
        past_key_values=None,
    ):
        '''
        ## Param:
        :param: hidden_states, shape = [n_layers, bz, seq_len, d]
        
        ## Return: 
            return BaseModelOutput( 
                last_hidden_state=hidden_states,  
                hidden_states=all_hidden_states,  # dtype = Tuple[Tensor] shape = [n_layers, bz, seq_len, d] 
                attentions=all_attentions, 
            )
        '''
        attention_mask = self.get_attention_mask(attention_mask) # 把掩码的形状规范化到4维 [bz, 1, seq_len, seq_len]
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos) # shape = [1, q, k]
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        
        # hidden_states 是当前层的隐藏状态，它可能是一个序列（例如包含多层的输出），也可能是一个单一的张量。
        if isinstance(hidden_states, Sequence):
            # hidden_states 可能包含了多层的输出，而我们只需要取第一层的输出作为下一层的输入。
            next_kv = hidden_states[0] # next_kv 主要用于存储下一个要处理的键值对, next_kv 存储了传递给下一层的输入
        else:
            next_kv = hidden_states # 表示当前层传进来的hidden_states 只是上一层的输出，并不包含再之前所有层的输出
        rel_embeddings = self.get_rel_embedding()
        
        for i, layer_module in enumerate(self.layer):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 里面存放了每一层的前缀 token embedding
            past_key_value = past_key_values[i] if past_key_values is not None else None

            hidden_states = layer_module.forward(
                next_kv,
                attention_mask,
                output_attentions,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
                past_key_value=past_key_value,  
            ) # shape = [bz, seq_len, d], 此时的 hidden_states 已经包含第0~i 层的所有隐状态
            
            
            if output_attentions:
                hidden_states, att_m = hidden_states

            if query_states is not None:
                query_states = hidden_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = hidden_states # 要传到下一个layer中的输入

            if output_attentions:
                all_attentions = all_attentions + (att_m,)
                
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
    
    
    
    
    


def build_relative_position(query_size, key_size, device):
    """
    Build relative position according to the query and key

    We assume the absolute position of query :math:`P_q` is range from (0, query_size) and the absolute position of key
    :math:`P_k` is range from (0, key_size), The relative positions from query to key is :math:`R_{q \\rightarrow k} =
    P_q - P_k`

    Args:
        query_size (int): the length of query
        key_size (int): the length of key

    Return:
        :obj:`torch.LongTensor`: A tensor with shape [1, query_size, key_size]
        
        
    ## Function:
    for example:
            q_ids = [
                    [0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2, 2],
                    [3, 3, 3, 3, 3, 3],
                    [4, 4, 4, 4, 4, 4],
                ]
        
        k_ids = [
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                    [0, 1, 2, 3, 4, 5],
                ]
                
        q_ids - k_ids = [
                    [0, 1, 2, 3, 4, 5],
                    [-1, 0, 1, 2, 3, 4],
                    [-2, -1, 0, 1, 2, 3],
                    [-3, -2, -1, 0, 1, 2],
                    [-4, -3, -2, -1, 0, 1],
                ] 
        ]
    """
    
    q_ids = torch.arange(query_size, dtype=torch.long, device=device)
    k_ids = torch.arange(key_size, dtype=torch.long, device=device)
    rel_pos_ids = q_ids[:, None] - k_ids.view(1, -1).repeat(query_size, 1) # shape = [q, k]
    rel_pos_ids = rel_pos_ids[:query_size, :]
    
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    
    return rel_pos_ids
    



@torch.jit.script
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    '''
    ##Args:
    c2p_pos: 相对位置索引，形状为 [1, 1, q_size, k_size]
            # relative_pos.shape = (1, 1, q_size, k_size)
            # query_layer.shape = [bz, num_heads, seq_len, head_size]

    将相对位置索引（c2p_pos）扩展成与 query_layer 的形状匹配的索引张量。
    
    
    ## return 
    扩展后的索引张量，形状为 [bz, num_heads, seq_len, k_size]
    '''
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])

@torch.jit.script
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    '''
    ##Args:
        c2p_pos: 相对位置索引，形状为 [1, 1, q_size, k_size]

        query_layer: 用于计算注意力的查询层，形状为 [bz, num_heads, seq_len, head_size]

        key_layer: 用于计算注意力的键层，形状为 [bz, num_heads, seq_len, head_size]
        
    ## return
    扩展后的索引张量，形状为 [bz, num_heads, key_len, key_len]

    '''
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])


@torch.jit.script
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    '''
    ##Args:
        pos_index: 相对位置索引，形状为 pos_idx.shape = (B, H, Q_LEN, 1)  

        p2c_att: 位置到内容注意力矩阵，形状为    [bz, num_heads, query_len, key_len, ]
                                        
        key_layer.shape = [bz, num_heads, key_len, head_size]
        
    ## return:
        扩展后的索引张量，形状为 [bz, num_heads, q_len, k_len]
    '''
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))
    

class DisentangledSelfAttention(nn.Module):
    """
    Disentangled self-attention module

    Parameters:
        config (:obj:`str`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            `BertConfig`, for more details, please refer :class:`~transformers.DebertaConfig`

    """
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # in_proj 将输入的特征投影到查询、键和值的空间中。
        self.in_proj = nn.Linear(config.hidden_size, self.all_head_size * 3, bias=False)
        
        # 偏置项创新点：仅对Q和V添加可学习偏置
        self.q_bias = nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
        self.v_bias = nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
        
        # 相对位置编码相关 
        # self.pos_att_type 用于指定使用哪种位置注意力机制
            # 一共有3种位置注意力机制：
            # "p2c": 位置到内容注意力机制，`将位置信息投影到内容空间中`，然后计算内容和位置之间的注意力。
            # "c2p": 内容到位置注意力机制，将内容信息投影到位置空间中，然后计算内容和位置之间的注意力。
            # "c2c": 内容到内容注意力机制，将内容信息投影到内容空间中，然后计算内容和内容之间的注意力。
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []

        self.relative_attention = getattr(config, "relative_attention", False)
        self.talking_head = getattr(config, "talking_head", False)

        if self.talking_head:
            self.head_logits_proj = nn.Linear(config.num_attention_heads, config.num_attention_heads, bias=False)
            self.head_weights_proj = nn.Linear(config.num_attention_heads, config.num_attention_heads, bias=False)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_dropout = StableDropout(config.hidden_dropout_prob)

            if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
                # pos_proj = W_{k,r} \in R^{d x d},
                self.pos_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
            if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
                # W_{q,r} \in R^{d x d}
                self.pos_q_proj = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = StableDropout(config.attention_probs_dropout_prob)
            
            
    def transpose_for_scores(self, x):
        '''
        x.shape = [bz, seq_len, hidden_size]
        
        ## return
        x.shape = (bz, num_heads, seq_len, head_size)
        '''
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1) # shape = [bz, seq_len, num_heads, head_size]
        
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # shape = (bz, num_heads, seq_len, head_size)
    
    
    
    def forward(
        self,
        hidden_states,
        attention_mask,
        return_att=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        past_key_value=None,
    ):
        """
        Call the module

        ## Args:
            hidden_states (:obj:`torch.FloatTensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                `Attention(Q,K,V)`

            attention_mask (:obj:`torch.ByteTensor`):
                An attention mask matrix of shape [`B`, `N`, `N`] where `B` is the batch size, `N` is the maximum
                sequence length in which element [i,j] = `1` means the `i` th token in the input can attend to the `j`
                th token.

            return_att (:obj:`bool`, optional):
                Whether return the attention matrix.

            query_states (:obj:`torch.FloatTensor`, optional):
                The `Q` state in `Attention(Q,K,V)`.

            relative_pos (:obj:`torch.LongTensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [`B`, `N`, `N`] with
                values ranging in [`-max_relative_positions`, `max_relative_positions`].

            rel_embeddings (:obj:`torch.FloatTensor`):
                The embedding of relative distances. It's a tensor of shape [:math:`2 \\times
                \\text{max_relative_positions}`, `hidden_size`].
                
                
                
            past_key_value.shape =  tuple(key_layer, value_layer), key_layer.shape = value_layer.shape = [bz, num_heads, prefix_seq_len, head_size]


        """
        
        if query_states is None:
            qp = self.in_proj(hidden_states)  # shape = [bz, seq_len, 3 * hidden_size]
            query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
            # query_layer.shape = [key_layer.shape] = [bz, num_heads, seq_len, head_size] 
        else:
            # in_proj.shape = [3 * num_heads, head_size]
            ws = self.in_proj.weight.chunk(self.num_attention_heads * 3, dim = 0) # shape = [3 * num_heads, head_size]
            # ws[0]
            qkvw = [torch.cat([ws[3*i + k] for i in range(self.num_attention_heads)], dim = 0) for k in range(3)] # shape = [3, num_heads, head_size]
            qkvb = [None] * 3
            
            q = self.linear(qkvw[0], qkvb[0], query_states) #  (num_heads, head_size) x (head_size, num_heads) = (num_heads, num_heads) 
            k, v = [self.linear(qkvw[i], qkvb[i], hidden_states) for i in range(1,3)]
            query_layer, key_layer, value_layer = [self.transpose_for_scores(x) for x in [q, k, v]]
            
            
        query_layer =  query_layer + self.transpose_for_scores(self.q_bias[None, None,:])
        value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :]) # shape = [bz, num_heads, seq_len, head_size]
        
        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1 + len(self.pos_att_type)
        scale = math.sqrt(query_layer.size(-1) * scale_factor) # sqrt(3x dk)

        past_key_value_length = past_key_value.shape[3] if past_key_value is not None else 0
        
        if past_key_value is not None:
            key_layer_prefix = torch.cat([past_key_value[0], key_layer], dim = 2) # shape = [bz, num_heads, past_key_value_length + seq_len, head_size]
            value_layer_prefix = torch.cat([past_key_value[1], value_layer], dim=2) # shape = [bz, num_heads, past_key_value_length + seq_len, head_size]
        else:
            key_layer_prefix = key_layer # 没有前缀的情况， key_layer_prefix.shape = key_layer.shape =  [bz, num_heads, seq_len, head_size]


        query_layer /= scale
        # 计算标准的注意力分数
        # query_layer.shape = [bz, num_heads, seq_len, head_size]
        attention_scores = torch.matmul(query_layer, key_layer_prefix.transpose(-1, -2))
        
        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            # query_layer.shape = [bz, num_heads, seq_len, head_size]
            rel_att = self.disentangled_att_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)
        
        if rel_att is not None:
            if past_key_value is not None:
                att_shape = rel_att.shape[:-1] + (past_key_value_length,)
                prefix_att = torch.zeros(*att_shape).to(rel_att)
                rel_att = torch.cat([prefix_att, rel_att], dim = -1)
                attention_scores += rel_att
            else:
                attention_scores += rel_att
                
        # shape = (bz, nheads, seqlen, d)
        if self.talking_head: # talk-head 机制： 在softmax前后，分别过一次投影
            attention_scores = self.head_logits_proj(attention_scores)
        
        softmax_mask = attention_mask[:,:,past_key_value_length:, :]
        
        attention_probs = XSoftmax.apply(attention_scores, softmax_mask, dim=-1)
        attention_probs = self.dropout(attention_probs) # shape = (bz, nheads, seqlen, seqlen)

        if self.talking_head:
            
            # a1 = attention_probs.permute(0,2,3,1).shape = (bz, seqlen, seqlen, nheads)
            #  a2 = (nheads, nheads) x a1.shape =  (bz, seqlen, seqlen, nheads)
            # a2.permute = (bz, nheads, seqlen, seqlen)
            attention_probs = self.head_weights_proj(attention_probs.permute(0,2,3,1)).permute(0,3,1,2)
        
        context_layer = torch.matmul(attention_probs, value_layer)  # shape = (bz, nheads, seqlen, d)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() 
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if return_att:
            return (context_layer, attention_probs)
        else:
            return context_layer
        
    
    def linear(self,w,b,x):
        '''
        w.shape = [bz, seq_len, hidden_size]
        b.shape = [bz, seq_len, hidden_size]
        x.shape = [bz, seq_len, hidden_size]
        '''
        if b is not None:
            return torch.matmul(x, w.t()) + b.t()
        else:
            return torch.matmul(x, w.t())
    
    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        '''
        ##Function:
        计算位置注意力的偏置项
        具体流程：
            1. 计算相对位置的编码
            2. 计算相对位置的注意力分数
            3. 计算注意力分数
            
        relative_pos：存储了 token i 和 token j 之间的相对位置
        rel_embeddings： 提供了相对位置 f(i-j) 的嵌入表示
        
        
        ## Args:
            query_layer.shape = key_layer.shape = value_layer.shape = [bz, num_heads, query_size, head_size]
            relative_pos.shape = [bz, seq_len, seq_len] or [1, query_size, key_size]
            rel_embeddings.shape = [2k, d]
            
        ##Return:
            相对位置的注意力分数: scores, scores.shape = [bz, num_heads, q_len, k_len]
        '''
        
        if relative_pos is None:
            q_len = query_layer.size(-2) # query_len
            k_len = key_layer.size(-2)
            relative_pos = build_relative_position(q_len, k_len, device = query_layer.device) # shape = [1, q_size, k_size]
    
        if relative_pos.dim()==2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
            
        elif relative_pos.dim()==3:
            relative_pos = relative_pos.unsqueeze(1)
            
        elif relative_pos.dim() != 4:
            raise ValueError(f"The shape of relative_pos must be 2 or 3 or 4. {relative_pos.dim()}")

        # 注意力跨度，即，模型能够感知的最大相对位置范围。
        att_span = min(max(query_layer.size(-2),key_layer.size(-2)), self.max_relative_positions) # <= max_relative_positions
        relative_pos = relative_pos.long().to(query_layer.device)
        rel_embeddings = rel_embeddings[
            self.max_relative_positions - att_span: self.max_relative_positions + att_span, :
        ].unsqueeze(0) # shape = [1, 2k, d], where att_span==k
        
        
        if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
            # 计算相对位置的编码 K_r
            pos_key_layer = self.pos_proj(rel_embeddings) # shape = [1, 2k, d] = [1, 2k, d] x [d, d]  K_r = P x W_{k,r}, where P = relative position embedding vectors, W_{k,r} = projection matrix
            pos_key_layer = self.transpose_for_scores(pos_key_layer) # shape = [1, 2k, num_heads, head_size]
        
        if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
            # 计算相对位置的注意力分数 Q_r
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer) # shape = [1, num_heads, 2k, head_size]
            
        
        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            # query_layer.shape = [bz, num_heads, q_len, head_size]
            # pos_key_layer.shape = [1, num_heads, 2k, head_size]
            # 计算 Q x R^T
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2)) # 广播乘法 shape = [bz, num_heads, q_len, 2k]

            # relative_pos.shape = (1, 1, q_size, k_size)
            # query_layer.shape = [bz, num_heads, q_len, head_size]
            # torch.clamp 是一个 PyTorch 函数，用于将张量中的值限制在指定范围内
            # 这里，将 relative_pos + att_span 的值限制在 [0, 2*att_span-1] 范围内
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1) # 将相对位置索引映射到一个非负范围
            
            # torch.gather 是 PyTorch 中的一个高级索引操作。
            # 它从指定维度（dim=-1）中，根据提供的 index 张量提取对应的值。
            # 这一行的作用是，从 c2p_att 中提取与动态位置索引（由 c2p_dynamic_expand 生成）对应的注意力分数。
            # 对应解耦注意力中 K_r的下标：delta(i,j)， 把c2p[i,j] 替换成 c2p[i][delta(i,j)}, delta就是index矩阵
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(c2p_pos, query_layer, relative_pos)) # index.shape = [bz, num_heads, q_size, k_size]
            
            # 上面这句代码的本质就是使用 rel_pos 的信息来重新分配  Q x R^T 矩阵中的值
            
            score += c2p_att

        # position->content
        if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
            # [1] 位置查询归一化  
            # 原始形状：pos_query_layer.shape = (B, H, P_LEN, D)  
            # 其中 P_LEN=2*att_span 是位置编码的总长度（左右两个方向）  
            pos_query_layer /= math.sqrt(scale_factor * query_layer.shape[-1] ) # sqrt(3d) shape = [1, num_heads, 2k, head_size]
            
            # [2] 构建新的相对位置矩阵（当Q≠K时）（如encoder-decoder结构）  
            if query_layer.size(-2) != key_layer.size(-2):
                # 获取query和key之间的相对位置矩阵
                r_pos = build_relative_position(
                    key_layer.size(-2), key_layer.size(-2), device = query_layer.device
                ) # shape = [1, key_size, key_size]
                # 示例：当K_LEN=5时，生成从-4到4的相对位置矩阵 
            else:
                r_pos = relative_pos # (B,H,Q,K)  
                
            # [3] 位置索引映射（镜像翻转） 
            # 原始r_pos范围：[-att_span, att_span] 
            p2c_pos = torch.clamp(-r_pos + att_span, 0, 2*att_span-1) # shape = [1, 1, k_size, k_size]
            # 操作解析：将位置i->j映射为j->i的位置编码  
            # 示例：当att_span=4时：  
            #   original pos: -4 → 8 (因为 -(-4)+4=8)  
            #   pos 2 → -2+4=2 → clamp后保持2  
            # 形状变化：保持与r_pos相同 (B, H, Q_LEN, K_LEN) 或 (1, K_LEN, K_LEN)  
            
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1) 
                # 分解动作：  
                    # relative_pos[:,:,:,0] → 取每个key位置的第0个相对位置 → (B, H, Q_LEN)  
                    # unsqueeze(-1) → (B, H, Q_LEN, 1)  
                    # 作用：为后续的gather操作准备索引  
        
        if "p2c" in self.pos_att_type:
            # key_layer.shape = [bz, num_heads, key_len, head_size]
            # pos_query_layer.shape = [1, num_heads, 2k, head_size] where 2k == P_LEN
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2)) # shape = # (B,H,K_LEN,P_LEN) 
            # 对应解耦注意力中 Q_r的下标：delta(j,i)， 把p2c[j,i] 替换成 p2c[j][delta(j,i)}, delta就是index矩阵

            p2c_att = torch.gather(
                # p2c_att.shape = (B,H,K_LEN,P_LEN)
                # p2c_pos.shape = (1, K_LEN, K_LEN)  
                # index.shape =  [bz, num_heads, key_len, key_len]
                p2c_att, dim=-1, index = p2c_dynamic_expand(p2c_pos, key_layer, relative_pos)
            ).transpose(-1,-2) # shape =  [bz, num_heads, key_len, key_len]
            
            # [3] 跨序列长度对齐
            if query_layer.size(-2) != key_layer.size(-2):
                # pos_idx.shape = (B, H, Q_LEN, 1)  
                # p2c_att.shape = [bz, num_heads, query_len, key_len]
                index = pos_dynamic_expand(pos_index, p2c_att, key_layer) # shape = [bz, n_heads, q_len, k_len]
                p2c_att = torch.gather(
                    p2c_att, dim=-2, index=index
                ) # shape = [bz, num_heads, q_len, k_len]
                score += p2c_att
        
        
        return score
        
        
class DebertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""       
    def __init__(self, config):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)
        
        self.position_biased_input = getattr(config, "position_biased_input", True)
        
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        if config.type_vocab_size>0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)
        
        
        if self.embedding_size != config.hidden_sizeq:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, eps = config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob) 
        self.config = config
        
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        
        
    def forward(
        self,
        input_ids = None,
        token_type_ids = None,
        position_ids = None,
        mask = None,
        inputs_embeds = None,
        past_key_values_length = 0,
    ):
        '''
        
        '''
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
            
            
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: past_key_values_length+ seq_length]
        
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device = self.position_ids.device)
        
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            
        
        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids)
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)
        
        
        embeddings = inputs_embeds
        
        if self.position_biased_input:
            embeddings+=position_embeddings
        if self.config.type_vocab_size >0:
            embeddings += self.token_type_embeddings(token_type_ids)
            
        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)
            
            
        embeddings = self.LayerNorm(embeddings)
        
        # embeddings 中的每个hidden unit都需要分配一个掩码
        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim()==4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2) # shape = (b, L, 1)
            mask = mask.to(embeddings.device)

            embeddings = embeddings * mask
        embeddings  = self.dropout(embeddings)
        return embeddings
        

        
        
        
class DebertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    
    config_class = DebertaConfig
    base_model_prefix = "deberta"
    _keys_to_ignore_on_load_missing = ["position_ids"] # 在加载预训练模型时，如果缺少 "position_ids" 这个键，将忽略该错误。
    _keys_to_ignore_on_load_unexpected = ["position_embeddings"] # 在加载预训练模型时，如果出现 "position_embeddings" 这个意外的键，将忽略该错误。
    
    def __init__(self, config):
        super().__init__(config)
        '''
        注册一个预加载钩子函数 self._pre_load_hook，
            该钩子函数会在加载模型状态字典之前被调用，用于对加载过程进行一些预处理。
        '''
        self._register_load_state_dict_pre_hook(self._pre_load_hook)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data._normal(mean=0.0, std=self.config.initializer_range)
        
            if module.bias is not None:
                module.bias.data._zero()
        
        elif isinstance(module, nn.Embedding):
            module.weight.data._normal(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx]._zero()
    
    
    
    def _pre_load_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        Removes the classifier if it doesn't have the correct number of labels.
        """
        # 从 state_dict 中移除 "classifier.weight" 和 "classifier.bias" 这两个键
        self_state = self.state_dict()
        
        if (
            ("classifier.weight" in self_state)
            and ("classifier.weight" in state_dict)
            and self_state["classifier.weight"].size() != state_dict["classifier.weight"].size()
        ): # 模型自身的分类器权重与外面传进来的分类器权重大小不一致
            logger.warning(
                f"The checkpoint classifier head has a shape {state_dict['classifier.weight'].size()} and this model "
                f"classifier head has a shape {self_state['classifier.weight'].size()}. Ignoring the checkpoint "
                f"weights. You should train your model on new data."
            )
            
            del state_dict["classifier.weight"]
            if "classifier.bias" in state_dict:
                del state_dict["classifier.bias"]
        



DEBERTA_START_DOCSTRING = r"""
    The DeBERTa model was proposed in `DeBERTa: Decoding-enhanced BERT with Disentangled Attention
    <https://arxiv.org/abs/2006.03654>`_ by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen. It's build on top of
    BERT/RoBERTa with two improvements, i.e. disentangled attention and enhanced mask decoder. With those two
    improvements, it out perform BERT/RoBERTa on a majority of tasks with 80GB pretraining data.

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.```


    Parameters:
        config (:class:`~transformers.DebertaConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

DEBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.DebertaTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.encode` and :func:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""

@add_start_docstrings(
    "The bare DeBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    DEBERTA_START_DOCSTRING,
)
class DebertaModel(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = DebertaEmbeddings(config)
        self.encoder = DebertaEncoder(config)
        self.z_steps = 0 # 
        self.config = config
        self.init_weights()
        
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings
    
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError("The prune function is not implemented in DeBERTa model.")

    
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        
        elif input_ids is not None:
            input_shape = input_ids.shape()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        
        # input_embedding 的 填充掩码 (只能约束前缀之后的token)
        embedding_mask = attention_mask[:, past_key_values_length:].contiguous()
        
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length+past_key_values_length), device = device)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            
        embedding_output = self.embeddings.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask = embedding_mask,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        
        encoder_outputs = self.encoder.forward(
            embedding_output,
            attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        
        
        encoded_layers:Tuple[torch.Tensor] = encoder_outputs[1] # 获取所有层的hidden_states
        
        '''
        实现一个多步（z_steps > 1）的递归计算，
        其中每一步使用 Transformer encoder 的最后一层（self.encoder.layer[-1]）对输入进行迭代处理
        
        这种机制可能是为了增强某些特定任务（如序列建模、生成任务、强化学习等）对输入特征的捕获能力。
        
        在多步计算中，每一步都可能基于前一步的输出进行更新，以逐渐优化特征表示。
            复制层的操作可能是为了在递归计算中重复利用相同的参数，从而节省模型的存储和计算开销

        通常，最后一层的输出（encoded_layers[-1]）用于最终预测，
        但在某些任务中，倒数第二层的输出（encoded_layers[-2]）也可能被用于进一步处理。
        '''
        if self.z_steps > 1:
            hidden_states = encoded_layers[-2]
            layers:List[DebertaLayer] = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            '''
            self.encoder.layer[-1] 是 Transformer 编码器的最后一层。
            通过复制这层多次（for _ in range(self.z_steps)），模型在后续步骤中可以多次使用这层进行递归计算。
            '''
            query_states  = encoded_layers[-1]
            rel_embeddings = self.encoder.get_rel_embedding()
            attention_mask  = self.encoder.get_attention_mask(attention_mask)
            rel_pos = self.encoder.get_rel_pos(embedding_output) # shape = (1, q_size, k_size)
            
            for layer in layers[1:]:
                query_states = layer.forward(
                    hidden_states,  
                    attention_mask,     
                    return_att=False,  
                    query_states=query_states,  
                    relative_pos=rel_pos,     # rel_pos 用于表示查询序列和键序列之间的相对位置，rel_embeddings 用于提供相对位置的嵌入表示。
                    rel_embeddings=rel_embeddings,  
                )
                encoded_layers.append(query_states)
                
                
                '''
                query_states 通常是查询矩阵 Q
                    在相对位置嵌入机制中，query_states 会叠加相对位置的偏置项 rel_embeddings，以增强查询矩阵的特征表示能力。

                rel_embeddings 通常是相对位置嵌入矩阵 R，
                    它包含了相对位置之间的关系信息，用于增强模型对相对位置的建模能力。

                    
                以下是标准注意力+相对位置嵌入：

                \[
                \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + QR^T}{\sqrt{d_k}}\right)V
                \]

                '''
        
        sequence_output = encoded_layers[-1]
        
        
        if not return_dict:
            return  (sequence_output,) + encoder_outputs[1 if output_hidden_states else 2:]

        return BaseModelOutput(
            last_hidden_state = sequence_output,
            hidden_states = encoder_outputs.hidden_states if output_hidden_states else None,
            attentions = encoder_outputs.attentions,
        )


@add_start_docstrings("""DeBERTa Model with a `language modeling` head on top. """, DEBERTA_START_DOCSTRING)
class DebertaForMaskedLM(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaModel(config)
        self.cls = DebertaOnlyMLMHead(config)

        self.init_weights()      
    
    def get_output_embeddings(self):
        '''
        获取 LM Head (词表分类头, 实际上就是一个线性层)
        '''
        return self.cls.predictions.decoder
    
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
     
        
        
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta.forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0] # last_hidden_state
        prediction_scores = self.cls(sequence_output) # shape=  (bz, seqlen, vocab_size)
        
        '''
        只有 [MASK] 位置的标签会被用于计算 loss
        
        在labels中， 除了 [MASK] token的其他位置，全部被标为 -100
        
         [MASK] 位置的处理
        在输入数据的预处理阶段，通常会将 [MASK] 位置的标签设置为实际的目标词索引，而非 [MASK] 的位置会被设置为 -100。因此：

        [MASK] 位置的标签值在 [0, vocab_size) 范围内，会参与 loss 计算。
        非 [MASK] 位置的标签值为 -100，会被 CrossEntropyLoss 忽略。
        '''
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss() # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        
        
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
        
# copied from transformers.models.bert.BertPredictionHeadTransform with bert -> deberta
class DebertaPredictionHeadTransform(nn.Module):
    '''
    本质就是一个 FFN
    '''
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)
    
    def forward(self, hidden_states):
        '''
        这就是一个FFN的forward pass
        '''
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states



# copied from transformers.models.bert.BertLMPredictionHead with bert -> deberta
class DebertaLMPredictionHead(nn.Module):
    '''
    ##Intro
        这实际上就是一个词表大小的分类器
    '''
    def __init__(self, config):
        super().__init__()
        self.transform = DebertaPredictionHeadTransform(config)
        
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.bias = nn.Parameter(torch.zeros(config.vocab_size)) # 每个token对应一个bias
        
        self.decoder.bias = self.bias
        
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
        





# copied from transformers.models.bert.BertOnlyMLMHead with bert -> deberta
class DebertaOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = DebertaLMPredictionHead(config) # 分类头

    def forward(self, sequence_output):
        '''
        sequence_output.shape = (bz, seq_len, d)
        '''
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores






@add_start_docstrings(
    """
    DeBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
class DebertaForSequenceClassification(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaModel(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim 

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.init_weights()
        
    
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta.forward(
            input_ids,
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0] # shape = (B, L, D)
        pooled_output = self.pooler(encoder_layer) # pooler 会自动取最后一个 token
        pooled_output = self.dropout(pooled_output) # shape = (B, D)
        logits = self.classifier(pooled_output) # shape = (B, num_labels)
        
        
        loss = None
        '''
        任务类型判断：
            如果 self.num_labels == 1，说明是一个回归任务。
            如果 labels 是一维向量或者最后一维的大小为 1，则是一个分类任务。
            否则，假设是一个多标签分类任务。
        
        根据任务类型，选择适当的损失函数并计算损失：
            回归任务使用均方误差损失（MSELoss）。
            分类任务使用交叉熵损失（CrossEntropyLoss）。
            多标签分类任务使用 LogSoftmax 和加权损失。
        '''
        if labels is not None:
            if self.num_labels==1:
                # regression
                loss_fn = nn.MSELoss()
            elif labels.dim()==1 or labels.size(-1)==1: # single-label classification
                # nonzero(): 返回张量中值为True的元素的索引
                label_index = (labels >= 0).nonzero() # shape = (sb, ) or (sb, 1), where sb is the number of non-negative labels,  and is <=  batch size
                labels = labels.long()
                if label_index.size(0)>0:
                    # gather 用于使用 label_index 作为索引，从 logits 中提取有效部分（即标签不为负的样本）。
                    # logits.shape  =(B, num_labels)
                    # index.shape = (B, num_labels)
                    labeled_logits = torch.gather(logits, 0, index=label_index.expand(label_index.size(0), logits.size(1)))  # shape =  (sb, num_labels)
                    labels = torch.gather(labels, 0, label_index.view(-1)) #shape = (sb, )
                    
                    '''
                    一句话说清gather的作用：
                        - 概括：
                            对于一个二维张量 tensor，gather 函数的作用是根据给定的索引 index，从 tensor 中提取相应的元素。
                        
                        - gather后的new_logits.shape = index.shape, 要做到这一点，我们需确保 logits.shape >= index.shape

                        - 如果 dim = 0:
                        - new_logits[i][j] = logits[index[i][j]][j]
                        
                        - 如果 index矩阵中的值全为1 (首先第一个index列就是全1，横向扩展后也是全1)，那么无论i取哪一行，new_logits[i][j] 最终都会被映射会 logits[1][j]
                        - 而此时，在我们的例子中， index矩阵中存放的是索引，而索引必不可能重复，而index矩阵中全为1， 因此index矩阵必定只有1行。
                        
                        - 所以，整句代码的作用相当于 把整个logits矩阵映射到了logits矩阵的第1行
                    '''
                    
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(labeled_logits.view(-1, self.num_labels), labels.view(-1))
                
                else:
                    # 如果所有标签均为负（无效标签），则直接返回零损失。
                    
                    loss =  torch.tensor(0).to(logits)
                
    
            else: # multi-label classification
                '''
                LogSoftmax数学公式：
                    - 对于样本 x 中的每个元素 xi，LogSoftmax 函数的计算公式如下：
                    - LogSoftmaxqq(xi) =  log(exp(xi) / Σj exp(xj))
                    - 其中，exp(xj) 表示 xj 的指数函数。
                    - 这个公式的含义是，对于每个样本，LogSoftmax 函数将每个元素的对数减去了该样本中所有元素的对数和。
                
                多标签分类loss计算：
                    - 对于多标签分类任务，通常使用 LogSoftmax 函数将模型的输出转换为概率分布。
                    - 然后，使用加权损失函数（例如交叉熵损失）来计算损失。
                    - 加权损失函数会根据每个样本的标签权重来调整损失，以平衡不同标签的重要性。
                    
                    Loss = -(1/N) Σ{i=1 to N} Σ{j=1 to C} wij * log_softmax(xi) * yi
                    - 其中，wij 是标签权重，yi 是样本 i 的标签，log_softmax(xi) 是 LogSoftmax 函数的输出。
                    - 这个公式的含义是，对于每个样本，我们计算每个标签的加权损失，并将它们相加得到最终的损失。
                '''
                
                # logits.shape = (B, num_labels)
                log_softmax = nn.LogSoftmax(dim=-1)
                loss = -(log_softmax(logits)* labels).sum(-1).mean() # shape = (B, )
        
        
        
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        
        return SequenceClassifierOutput(
            loss = loss,
            logits = logits,
            hidden_states= outputs.hidden_states,
            attentions = outputs.attentions
        )
        
        
        
        
        




        
@add_start_docstrings(
    """
    DeBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
class DebertaForTokenClassification(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        for param in self.deberta.parameters():
            param.requires_grad = False

        self.init_weights()
        
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta.forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output) # shape = (bz, seqlen, vocab_len)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            # attention_mask 用于忽略填充位置的损失计算。
            # active_loss 是一个布尔张量，指示哪些位置是有效的（值为 1）。
            # active_logits 和 active_labels 分别是有效位置的预测值和真实值，用于计算损失。
            
            if attention_mask is not None: # 如果提供了 attention_mask，则只计算有效位置的损失
                # attention_mask.shapew = (bz, seqlen)
                active_loss = attention_mask.view(-1) == 1 # shape = (bz*seqlen, )
                active_logits = logits.view(-1, self.num_labels) #shape = (bz*seqlen, n_labels)
                # where: 如果condition成立，就返回第一个tensor中的值，否则返回第二个tensor中的值
                active_labels =  torch.where(
                    condition = active_loss, input = labels.view(-1), other = torch.tensor(loss_fct.ignore_index).type_as(labels)
                ) # (bz*seqlen,)
            
                loss = loss_fct(active_logits, active_labels)
            
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
         
        
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )







@add_start_docstrings(
    """
    DeBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    DEBERTA_START_DOCSTRING,
)
class DebertaForQuestionAnswering(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels) # num_labels 一般取2， 一个start, 一个 end

        self.init_weights()
        
        
        
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        start_positions=None, # label 1
        end_positions=None,  # label 2
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        
        outputs = self.deberta.forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0] # shape = (B, L, D)

        logits = self.qa_outputs(sequence_output) # shape = (B, L, 2)
        start_logits, end_logits = logits.split(1, dim=-1) # shape = (B, L ,1)
        
        start_logits = start_logits.squeeze(-1).contiguous() # shape = (B, L), 每个token所在的位置都对应着一个 0-1的logit， 代表该token是 start/end 的概率
        end_logits = end_logits.squeeze(-1).contiguous()
        
        
        total_loss = None
        
        if start_positions is not None and end_positions is not None:
            # start_positions.shape = (B, )
            # end_postions.shape = (B, )
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1: # e.g. (n_gpus, B, )
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1: # e.g. (n_gpus, B, )
                end_positions = end_positions.squeeze(-1) # (n_gpus * B, )
            
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            # start/end positions 超出序列长度
            
            ignored_index = start_logits.size(1) # seq_len
            # clamp 用来限定范围
            start_positions = torch.clamp(start_positions, 0, ignored_index)
            end_positions = torch.clamp(end_positions, 0, ignored_index)
            
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            
            total_loss = (start_loss + end_loss) / 2
            
        
        
        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss, ) + output) if total_loss is not None else output
        
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        