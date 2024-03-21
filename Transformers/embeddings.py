import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model, input_vocab_size):
        super().__init__()
        self.d_model = d_model
        self.input_vocab_size = input_vocab_size
        self.embeddings = nn.Embedding(input_vocab_size, d_model)

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)  
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_length, dropout) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(dropout)

        pos_arr = torch.zeros(seq_length, d_model)
        pos = torch.arange(0, seq_length, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_arr[:, 0::2] = torch.sin(pos * div_term)
        pos_arr[:, 1::2] = torch.cos(pos * div_term)

        pos = pos.unsqueeze(0)
        self.register_buffer('pos_arr', pos_arr)

    def forward(self, x):
        x = x + (self.pos_arr[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)