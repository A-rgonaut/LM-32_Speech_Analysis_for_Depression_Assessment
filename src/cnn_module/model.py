import torch
from torch import nn

from .config import CNNConfig

class CNNModel(nn.Module):
    def __init__(self, config: CNNConfig):
        super(CNNModel, self).__init__()
        self.config = config
        
        # conv1
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.config.dropout_rate),
            nn.BatchNorm1d(16)
        )
        
        # conv2
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=32, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.config.dropout_rate),
            nn.BatchNorm1d(32)
        )
        
        # conv3
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.config.dropout_rate),
            nn.BatchNorm1d(64)
        )
        
        # Global Average Pooling and Flatten
        self.glob_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        # MLP
        self.mlp_block = nn.Sequential(
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(128, 1)
        )

        self.init_weights()

    def init_weights(self):
        # initialize weights of classifier
        for name, param in self.named_parameters():
            if "weight" in name and len(param.shape) > 1:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        if isinstance(x, dict):
            x = x['input_values']

        if not torch.is_floating_point(x):
            x = x.float()

        # conv blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = self.glob_avg_pool(x)               
        x_flattened = self.flatten(x)           
        output = self.mlp_block(x_flattened)    

        return output