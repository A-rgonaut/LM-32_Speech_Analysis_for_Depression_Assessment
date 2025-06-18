import torch.nn as nn

class CNNMLP(nn.Module):
    def __init__(self, dropout_rate=0.25):
        super(CNNMLP, self).__init__()
        
        # conv1
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(16)
        )
        
        # conv2
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=32, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(32)
        )
        
        # conv3
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(64)
        )
        
        # Global Average Pooling and Flatten
        self.glob_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        # MLP
        self.mlp_block = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.init_weights()

    def init_weights(self):
        # initialize weights of classifier
        for name, param in self.named_parameters():
            if "weight" in name and len(param.shape) > 1:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, batch):
        x = batch['input_values']
        # Input: x.shape -> [batch_size, 1, 4000] (1 canale, 250ms a 16kHz)
        x = self.conv_block1(x)
        # Dopo Conv1d: [batch_size, 16, 4000 - 64 + 1] = [batch_size, 16, 3937]
        # Dopo MaxPool1d: [batch_size, 16, 3937 // 2] = [batch_size, 16, 1968]
        x = self.conv_block2(x)
        # Dopo Conv1d: [batch_size, 32, 1968 - 32 + 1] = [batch_size, 32, 1937]
        # Dopo MaxPool1d: [batch_size, 32, 1937 // 2] = [batch_size, 32, 968]
        x = self.conv_block3(x)
        # Dopo Conv1d: [batch_size, 64, 968 - 16 + 1] = [batch_size, 64, 953]
        # Dopo MaxPool1d: [batch_size, 64, 953 // 2] = [batch_size, 64, 476]
        x = self.glob_avg_pool(x) # [batch_size, 64, 1]
        x_flattened = self.flatten(x) # [batch_size, 64 * 1] = [batch_size, 64]
        output = self.mlp_block(x_flattened)
        return output