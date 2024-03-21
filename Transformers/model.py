import torch
import torch.nn as nn
import math
from encoder import Encoder, EncoderBlock
from decoder import Decoder, DecoderBlock
from embeddings import InputEmbeddings, PositionalEncoding
from utils import LinearLayer, FeedForward
from attention import MultiHeadAttention

class Transformers(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, input_embedding: InputEmbeddings, output_embedding: InputEmbeddings,
                 ip_pos_encoding: PositionalEncoding, op_pos_encoding: PositionalEncoding, prob_layer: LinearLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_embedding = input_embedding
        self.output_embedding = output_embedding
        self.ip_pos_encoding = ip_pos_encoding
        self.op_pos_encoding = op_pos_encoding
        self.prob_layer = prob_layer

    def encode(self, src, src_mask):
        src = self.input_embedding(src)
        src = self.ip_pos_encoding(src)
        src = self.encoder(src, src_mask)
        return src
    
    def decode(self, encoder_output, target, src_mask, target_mask):
        target = self.output_embedding(target)
        target = self.op_pos_encoding(target)
        target = self.decoder(target, encoder_output, src_mask, target_mask)
        return target
    
    def projection(self, x):
        return self.prob_layer(x)
    
def build_transformer(self, ip_vocab_size, op_vocab_size, ip_seq_len, op_seq_len, d_model = 512, N = 6, h = 8,
                      d_ff = 2048, dropout = 0.1):
    ip_embedding = InputEmbeddings(d_model, ip_vocab_size)
    op_embedding = InputEmbeddings(d_model, op_vocab_size)
    ip_pos_encoding = PositionalEncoding(d_model, ip_seq_len, dropout)
    op_pos_encoding = PositionalEncoding(d_model, op_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_attention_block = MultiHeadAttention(d_model, h, dropout)
        encoder_ff_block = FeedForward(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_attention_block, encoder_ff_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_ff_block = FeedForward(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_attention_block, decoder_cross_attention_block, decoder_ff_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection = LinearLayer(d_model, op_vocab_size)

    transformer = Transformers(encoder, decoder, ip_embedding, op_embedding,
                               ip_pos_encoding, op_pos_encoding, projection)
    
    for val in transformer.parameters():
        if val.dim > 1:
            nn.init.xavier_uniform_(val)

    return transformer