import torch
import torch.nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss

from transformers import BertModel, BertPreTrainedModel
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from model.prefix_encoder import PrefixEncoder
from model.deberta.deberta import DebertaModel, DebertaPreTrainedModel
# from model.debertaV2 import DebertaV2Model, DebertaV2PreTrainedModel

from model.chatglm2.modeling_chatglm2 import ChatGLMModel, ChatGLMPreTrainedModel




class DebertaPrefixForTokenClassification(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

        for param in self.deberta.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        deberta_param = 0
        for name, param in self.deberta.named_parameters():
            deberta_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - deberta_param
        print('total param (prefix encoder + classifier) is {}'.format(total_param)) # 9860105

    def get_prompt(self, batch_size):
        prefix_tokens = torch.arange(self.pre_seq_len).long()
        prefix_tokens = prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.deberta.device)
        
        past_key_values = prefix_tokens.view(
            batch_size,
            self.pre_seq_len,
            2*self.n_layer,
            self.n_head,
            self.n_embd
        )
        
        past_key_values = self.dropout(past_key_values)
        # 重排+分组
        past_key_values = past_key_values.permute([2,0,3,1,4]).split(2)
        return past_key_values

    
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        inputs_embeds = None,
        labels = None,
        return_dict=None,
        head_mask = None,
        output_attentions = None,
        output_hidden_states = None,
        ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        
        # 获取前缀的 KV head
        past_key_values = self.get_prompt(batch_size = batch_size)
        
        # 前缀的padding mask
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.deberta.device)
        # 完整的padding mask
        attention_mask = torch.concat((past_key_values, attention_mask), dim=1)
        
        outputs = self.deberta(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        
        sequence_output = outputs[0] # shape = [batch_size, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output) # shape = [batch_size, seq_len, num_labels]
        
        attention_mask = attention_mask[:, self.pre_seq_len:]

        loss = None
        
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            
            if attention_mask is not None:
                #  attention_mask.view(-1).shape = (batch_size * seq_len, )
                active_loss = attention_mask.view(-1) == 1  # 1 表示需要计算loss
                active_logits = logits.view(-1, self.num_labels) # shape = (bz * seqlen, num_labels)

                # labels.shape = (bz * seqlen, )
                active_labels = torch.where(
                    active_loss, input=labels.view(-1), other=torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                
                loss = loss_fct(active_logits, active_labels)
                    
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        
        
        
        if not return_dict:
            output = (logits,) + output[2:]
            return ((loss,)+ output) if loss is not None else output
        
        
        
        return TokenClassifierOutput(
            loss = loss,
            logits = logits,
            hidden_states = outputs.hidden_states,
            attentions = outputs.attentions,
            
        )





class ChatGLM2PrefixForTokenClassification():
    
    def __init__(self):
        pass