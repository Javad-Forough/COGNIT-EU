import torch
import torch.nn as nn

# Define the Feed-Forward Neural Network (FFNN) model
class FFNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FFNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()  # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Train function for FFNN
def train_ffnn_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Flatten inputs for FFNN
            inputs = inputs.view(inputs.size(0), -1)  # Flattening input for FFNN

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets[:, -1, :])

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')
