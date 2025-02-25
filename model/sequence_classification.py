import torch
from torch._C import NoopLogger
import torch.nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from transformers import BertModel, BertPreTrainedModel
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput, Seq2SeqLMOutput

from model.prefix_encoder import PrefixEncoder
from model.deberta.deberta import DebertaModel, DebertaPreTrainedModel, ContextPooler, StableDropout

from model.chatglm2.modeling_chatglm2 import ChatGLMModel, ChatGLMPreTrainedModel

import copy





class RobertaPrefixForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

        for param in self.roberta.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('total param is {}'.format(total_param)) # 9860105

    
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
    

    
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        
        
        outputs = self.roberta.forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype==torch.long or labels.dtype==torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type == 'multi_label_classification'
            
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels==1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        
        if not return_dict:
            output = (logits)  + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        
        
        return SequenceClassifierOutput(
            loss=loss,
            logits = logits,
            hidden_states = outputs.hidden_states,
            attentions = outputs.attentions 
        )














class RobertaPromptForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.embeddings = self.roberta.embeddings
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        for param in self.roberta.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)
    
    def get_prompt(self, batch_size):
        '''
        ##return
            return prompts, shape = (batch_size, hidden_size)
        '''
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts
    
    
    
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        raw_embedding = self.embeddings(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
        )
        
        prompts = self.get_prompt(batch_size=batch_size) # (bz, hz)
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.roberta.device)

        attention_mask = torch.cat([prefix_attention_mask, attention_mask],dim=1)
        
        outputs = self.roberta(
            # input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # past_key_values=past_key_values,
        )
        
        
        sequence_output = sequence_output[0]
        sequence_output = sequence_output[:,self.pre_seq_len:, :].contiguous()
        
        # get the </s> token 
        first_token_tensor = sequence_output[:,0] # shape = (bz, hz)
        pooled_output = self.roberta.pooler.dense(first_token_tensor)
        pooled_output = self.roberta.pooler.activation(pooled_output)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output) # shape = (bz, n_labels)
        
        
        loss = None
        if labels is not None:
            
            if self.config.problem_type is None:
                if self.num_labels==1:
                    self.config.problem_type=="regression"
                
                elif self.num_labels>1 and (labels.dtype==torch.long or labels.dtype==torch.int):
                    self.config.problem_type=="single_label_classification"
                else:
                    self.config.problem_type == "multi_label_classification"
            
            if self.config.problem_type=="regression":
                loss_fct = MSELoss()
                
                if self.num_labels ==1:
                    # logits.squeeze().shape = (bz*n_labels, )
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
                    '''
                    在多分类问题中，每个样本的真实标签通常是一个独热编码（One-Hot Encoding）向量，预测值是一个概率分布向量。因此，MSE损失的计算可以按照以下步骤进行：

                        1. 计算每个样本的预测值与真实标签之间的差值。
                        2. 对差值进行平方。
                        3. 对所有样本的平方差值求和。
                        4. 除以样本数量得到平均损失。
                    '''
            
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                # 由于只取第一个token，变换后的 logits.shape = (bz, n_labels), labels.shape = (bz,)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                
            elif self.config.problem_type=="multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits,labels)
        
        
        if not return_dict:
            output =  (logits, )+ output[2:]
            return ((logits,) + output)  if logits is not None else output
        
        

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions = outputs.attentions
        )

class DebertaPrefixForSequenceClassification(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.deberta = DebertaModel(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        self.classifier = torch.nn.Linear(output_dim, self.num_labels)
        self.dropout = StableDropout(config.hidden_dropout_prob)
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
        print('total param is {}'.format(total_param)) # 9860105
        
        
        
        
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.deberta.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2, 
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
    
    
    
    def forward(
        self
        ):
        pass

        
        
        
        



class ChatGLM2PrefixForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.chatglm2 = ChatGLMModel(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim
        self.classifier = torch.nn.Linear(output_dim, self.num_labels)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.init_weights()

        for param in self.chatglm2.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        chatglm2_param = 0
        for name, param in self.chatglm2.named_parameters():
            chatglm2_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - chatglm2_param
        print('total param is {}'.format(total_param)) # 9860105






class ChatGLM2PromptForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.chatglm2 = ChatGLMModel(config)
        self.embeddings = self.chatglm2.embeddings
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        for param in self.chatglm2.parameters():
            param.requires_grad = False
        
        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)
    
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.chatglm2.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts