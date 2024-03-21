import torch
import torch.nn as nn

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

class LinearLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        self.prob = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.prob(x), dim = -1)