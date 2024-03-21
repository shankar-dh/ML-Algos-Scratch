import torch
import torch.nn as nn
from attention import MultiHeadAttention
from utils import FeedForward, ResidualConnection, LayerNormalization

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
