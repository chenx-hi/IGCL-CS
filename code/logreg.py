import torch
import torch.nn as nn
import torch.nn.functional as F


class LogReg(nn.Module):
    def __init__(self, hidden_channels, num_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        out = self.fc(x)
        return out

# class LogReg(nn.Module):
#     def __init__(self, input_dim, output_dim, num_layers, dropout):
#         super(LogReg, self).__init__()
#         self.linears = torch.nn.ModuleList()
#         self.linears.append(nn.Linear(input_dim, output_dim))
#         self.dropout = dropout
#         for layer in range(num_layers - 1):
#             self.linears.append(nn.Linear(output_dim, output_dim))
#         self.num_layers = num_layers
#
#     def forward(self, embedding):
#         h = embedding
#         for layer in range(self.num_layers - 1):
#             h = self.linears[layer](h)
#         h = F.dropout(h, p=self.dropout, training=self.training)
#         h = self.linears[self.num_layers - 1](h)
#         return h
