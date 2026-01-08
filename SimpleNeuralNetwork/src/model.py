utf-8import torch
import torch.nn as nn
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, dropout_rate=0.3):
        super(SimpleNN, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)