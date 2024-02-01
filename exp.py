import torch
from torch import nn

features = torch.rand((1, 3, 1024))
num_token = features.size(1)
if num_token>1:
    features = features.permute(0,2,1)
    m = nn.MaxPool1d(kernel_size = num_token, stride=10000)
    features = m(features)
    features = features.permute(0,2,1)