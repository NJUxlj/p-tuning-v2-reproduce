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
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])


@torch.jit.script
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
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
                att_shape = None
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
        ## Args:
            query_layer.shape = key_layer.shape = value_layer.shape = [bz, num_heads, seq_len, head_size]
            relative_pos.shape = [bz, seq_len, seq_len]
            rel_embeddings.shape = [2k, d]
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
            pos_q_layer = self.pos_q_proj(rel_embeddings)
            pos_q_layer = self.transpose_for_scores(pos_q_layer) # shape = [1, 2k, num_heads, head_size]
            
        
        score = 0
        # content->position
        if "c2p" in self.pos_att_type:
            # query_layer.shape = [bz, num_heads, seq_len, head_size]
            # pos_key_layer.shape = [1, 2k, num_heads, head_size]
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2)) # 广播乘法 shape = [max(bz, 1), max(num_heads, 2k), seq_len, num_heads]

            # relative_pos.shape = (1, 1, q_size, k_size)
            # query_layer.shape = [bz, num_heads, seq_len, head_size]
            # torch.clamp 是一个 PyTorch 函数，用于将张量中的值限制在指定范围内
            # 这里，将 relative_pos + att_span 的值限制在 [0, 2*att_span-1] 范围内
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1) # 将相对位置索引映射到一个非负范围
            
            # torch.gather 是 PyTorch 中的一个高级索引操作。
            # 它从指定维度（dim=-1）中，根据提供的 index 张量提取对应的值。
            # 这一行的作用是，从 c2p_att 中提取与动态位置索引（由 c2p_dynamic_expand 生成）对应的注意力分数。
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(c2p_pos, query_layer, relative_pos))
            score += c2p_att

        # position->content
        if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
            pass
        
        
        if "p2c" in self.pos_att_type:
            pass
        
        
        
        return score
        
        
class DebertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""       
    def __init__(self, config):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)
        
        
        
        
        
        
        
        
        
        
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
        self.z_steps = 0
        self.config = config
        self.init_weights()





@add_start_docstrings("""DeBERTa Model with a `language modeling` head on top. """, DEBERTA_START_DOCSTRING)
class DebertaForMaskedLM(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaModel(config)
        self.cls = DebertaOnlyMLMHead(config)

        self.init_weights()        
        
        
# copied from transformers.models.bert.BertPredictionHeadTransform with bert -> deberta
class DebertaPredictionHeadTransform(nn.Module):
    pass



# copied from transformers.models.bert.BertLMPredictionHead with bert -> deberta
class DebertaLMPredictionHead(nn.Module):
    pass





# copied from transformers.models.bert.BertOnlyMLMHead with bert -> deberta
class DebertaOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = DebertaLMPredictionHead(config)

    def forward(self, sequence_output):
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
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()