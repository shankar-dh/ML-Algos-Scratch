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
    
class LayerNormalization(nn.Module):

    def __init__(self, epsilon = 10**-7) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.epsilon) + self.bias

class FeedForward(nn.Module):

    def __init__(self, d_model, dff, dropout) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, dff) 
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
    
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, head, dropout):
        super().__init__()
        self.d_model = d_model 
        self.head = head 
        assert d_model % head == 0, "d_model and head should be divisible"

        self.d_k = d_model // head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_mask(mask ==0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # Q' (batch_size, seq_length, d_model)
        key = self.w_k(k)   # K' (batch_size, seq_length, d_model)
        value = self.w_v(v) # V' (batch_size, seq_length, d_model)

        #Splitting each vector into h heads 512/8 = 64 (batch_dim, seq_length, 8, 64) --> (batch_dim, 8, seq_length, 64)
        query = query.view(query.shape[0], query.shape[1], self.head, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.head, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.head, self.d_k).transpose(1,2)

        x, attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        x = x .transpose(1,2).contiguous.view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)
    

class ResidualConnection(nn.Module):
    def __init__(self, dropout) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sub_layer):
        input = x
        x = self.norm(x)
        x = sub_layer(x)
        x = self.dropout(x)
        x = input + x
        return x
    
class EncoderBlock(nn.Module):

    def __init__(self, attention_instance: MultiHeadAttention, ff_instance: FeedForward, dropout):
        super().__init__()
        self.attention_instance = attention_instance
        self.ff_instance = ff_instance
        self.attention_residual = ResidualConnection(dropout)
        self.ff_residual = ResidualConnection(dropout)

    def forward(self, x, src_mask):
        attention_output = self.attention_instance(x, x, x, src_mask)
        x = self.attention_residual(x, attention_output)
        ff_output =self.ff_instance(x)
        x = self.ff_residual(x, ff_output)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers 
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
    

class DecoderBlock(nn.Module):
    def __init__(self, attention_instance: MultiHeadAttention, cross_attention_instance: MultiHeadAttention, ff_instance: FeedForward, dropout):
        super().__init__()
        self.attention_instance = attention_instance
        self.cross_attention_instance = cross_attention_instance
        self.ff_instance = ff_instance
        self.attention_residual = ResidualConnection(dropout)
        self.cross_attention_residual = ResidualConnection(dropout)
        self.ff_residual = ResidualConnection(dropout)

    def forward(self, x, encoder_output, src_mask, target_mask):
        attention_output = self.attention_instance(x, x, x, target_mask)
        x = self.attention_residual(x, attention_output)
        cross_attention_output = self.cross_attention_instance(x, encoder_output, encoder_output, src_mask) #src_mask used for masking the padding tokens in the encoder i/p
        x = self.cross_attention_residual(x, cross_attention_output)
        ff_output = self.ff_instance(x)
        x = self.ff_residual(x, ff_output)

        return x 

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers 
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output,  src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)

        

class LinearLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        self.prob = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.prob(x), dim = -1)





