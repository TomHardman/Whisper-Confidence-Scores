import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class ConfidenceEstimationModel:
    def __init__(self):
        super().__init__()
        
        # Define layers with specified input and output sizes
        self.layer1 = nn.Linear(2305, 1024)
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