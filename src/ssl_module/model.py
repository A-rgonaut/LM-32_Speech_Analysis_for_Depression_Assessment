
import torch
from torch import nn
from config import CNNConfig

class SSLModel(nn.Module):

    def __init__(self, config: CNNConfig):
        self.config = config
        # TODO: Define the architecture of the model

    def init_weights(self):
        pass

    def forward(self, x):
        pass
