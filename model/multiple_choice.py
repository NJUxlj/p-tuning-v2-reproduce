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