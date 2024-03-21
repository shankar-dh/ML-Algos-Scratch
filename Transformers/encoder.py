import torch
import torch.nn as nn
from attention import MultiHeadAttention
from utils import FeedForward, ResidualConnection, LayerNormalization

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