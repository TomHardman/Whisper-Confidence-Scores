import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd 
import numpy as np


class SimpleCEM(nn.Module):
    def __init__(self, n: int, no_dec: bool = False, return_logits: bool = False):
        super().__init__()
        if no_dec:
            inp_size = 1537
        else:
            inp_size = 2305
        
        # Define layers with specified input and output sizes
        self.fc_layer = nn.Linear(inp_size, n)
        self.output_layer = nn.Linear(n, 1)
        self.return_logits = return_logits

        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sm_probs = torch.log(x[:, -1:])
        x = torch.cat((x[:, :-1], sm_probs), dim=1)

        # Forward pass through each layer
        x = torch.relu(self.fc_layer(x))
        logits = self.output_layer(x)

        if self.return_logits:
            return logits
        
        # Apply sigmoid activation to get the final output
        output = self.sigmoid(logits)
        
        return output


class TemperatureCalibrator(nn.Module):
    def __init__(self, whisper_model):
        super().__init__
        self.linear = whisper_model.decoder.token_embedding
        self.single_unit = nn.Linear()
    
    def forward(self, x):
        x = (x @ torch.transpose(self.linear.weight.to(x.dtype), 0, 1)).float()




class CEMSkip(nn.Module):
    def __init__(self, n: int, no_dec: bool = False):
        super().__init__()
        if no_dec:
            inp_size = 1537
        else:
            inp_size = 2305
        
        # Define layers with specified input and output sizes
        self.fc_layer = nn.Linear(inp_size, n)
        self.output_layer = nn.Linear(n + 1, 1)

        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sm_probs = torch.log(x[:, -1:])

        # Forward pass through each layer
        x = torch.cat((torch.relu(self.fc_layer(x)), sm_probs), dim=1)
        logits = self.output_layer(x)
        
        # Apply sigmoid activation to get the final output
        output = self.sigmoid(logits)
        
        return output


class ConfidenceEstimationModulev2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define layers with specified input and output sizes
        self.layer1 = nn.Linear(2304, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.output_layer = nn.Linear(512, 1)  # Output layer with a single neuron

        # Sigmoid activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through each layer
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        logits = self.output_layer(x)
        
        # Apply sigmoid activation to get the final output
        output = self.sigmoid(logits)
        
        return output


class ConfidenceDataset(Dataset):
    def __init__(self, pickle_path, no_dec=False):
        self.df = pd.read_pickle(pickle_path)
        attn_features = self.df.iloc[:, 1:769].values
        emb_features = self.df.iloc[:, 1537:2305].values
        sm_probs = self.df.iloc[:, -2:-1].values

        if no_dec:
            self.features = np.concatenate((attn_features, emb_features, sm_probs), axis=1)
        else:
            self.features = self.df.iloc[:, 1:-1].values
        self.targets = self.df.iloc[:, -1].values
        self.num_samples = len(self.df)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0)
        return features, target
    
    def get_x_shape(self):
        return self.features.shape
    
    def get_y_shape(self):
        return self.targets.shape