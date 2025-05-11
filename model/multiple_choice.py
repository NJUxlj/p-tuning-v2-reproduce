import torch
from torch._C import NoopLogger
import torch.nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from transformers import BertModel, BertPreTrainedModel
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import MultipleChoiceModelOutput, BaseModelOutput, Seq2SeqLMOutput

from model.prefix_encoder import PrefixEncoder
from model.deberta.deberta import DebertaModel, DebertaPreTrainedModel, ContextPooler, StableDropout


from model.chatglm2.modeling_chatglm2 import ChatGLMModel,ChatGLMPreTrainedModel




class DebertaPrefixForMultipleChoice(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.deberta = DebertaModel(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        self.classifier = torch.nn.Linear(output_dim, 1)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.init_weights()
        
        for param in self.deberta.parameters():
            param.requires_grad = False
            
        self.pre_seq_len = config.pre_seq_len
        
        self.n_layer = config.num_hidden_layers
        
        self.n_head = config.num_attention_heads
        
        # head_size
        self.n_embd = config.hidden_size // config.num_attention_heads
        
        
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        deberta_param = 0
        
        
        
    def get_prompt(self, batch_size):
        pass
    
    
    
    
    
    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):
        pass
        
        







class ChatGLM2PrefixForMultipleChoice(ChatGLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.deberta = ChatGLMModel(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        self.classifier = torch.nn.Linear(output_dim, 1)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.init_weights()
        
        
    def get_prompt(self, batch_size):
        pass
    
    

    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):
        pass
            
            
            
        