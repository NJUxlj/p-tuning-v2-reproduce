import torch

'''



          
这份代码实现了一个名为`PrefixEncoder`的PyTorch模块，主要用于处理前缀编码任务。以下是它的核心功能解析：

1. **基本作用**：
- 这是一个继承自`torch.nn.Module`的神经网络模块
- 用于将输入的前缀序列编码为特定形状的张量
- 输出形状为`(batch-size, prefix-length, 2*layers*hidden)`

2. **两种编码模式**：
```python:d:/ai-code/p-tuning-v2-reproduce/model/prefix_encoder.py
if self.prefix_projection:
    # 使用两层MLP进行编码
    self.embedding = torch.nn.Embedding(...)
    self.trans = torch.nn.Sequential(...)
else:
    # 直接使用Embedding层
    self.embedding = torch.nn.Embedding(...)
```

3. **关键组件**：
- `prefix_projection`：控制是否使用投影机制
- `embedding`层：将离散的前缀标记转换为连续向量
- `trans`模块（当启用投影时）：包含线性层和Tanh激活函数的两层MLP

4. **典型应用场景**：
这种结构常用于前缀调优(P-Tuning)中，
为语言模型生成可训练的前缀表示。
输出通常会被用作Transformer模型中的past_key_values。



'''

class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    ''' 
    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values