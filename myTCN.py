import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Adjust padding to maintain input-output size match
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, 
                               padding=((kernel_size - 1) * dilation) // 2, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, 
                               padding=((kernel_size - 1) * dilation) // 2, dilation=dilation)

        # If the input and output dimensions don't match, use a 1x1 conv to adjust them
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(out + x)  # Residual connection

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size-1) * dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Example usage:
# model = TCN(num_inputs=1, num_channels=[16, 32], kernel_size=3, dropout=0.2)
