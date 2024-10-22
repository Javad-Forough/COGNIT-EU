import torch
import torch.nn as nn
import torch.nn.functional as F

# TCN Block with causal convolutions and dilations
class TemporalBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_size, output_size, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(output_size, output_size, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.downsample = nn.Conv1d(input_size, output_size, 1) if input_size != output_size else None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Residual connection
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(out + x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
    
    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_size)
        # TCN expects: (batch_size, input_size, sequence_length), so permute the dimensions
        x = x.permute(0, 2, 1)
        out = self.network(x)
        # Take the last time step's output for prediction
        out = out[:, :, -1]
        return self.linear(out)
