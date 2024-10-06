import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from data import load_data, scale_data, create_sequences
from model import LSTMModel, train_model

# Hyperparameters
input_size = 4
hidden_size = 64
num_layers = 2
output_size = 4  # Predicting the next 4 time steps
sequence_length = 96
prediction_length = 4
batch_size = 64
learning_rate = 0.001
num_epochs = 2

# Load and preprocess data
filename = 'workload_series.npy'
data_reshaped = load_data(filename)
data_scaled, scaler = scale_data(data_reshaped)
X, y = create_sequences(data_scaled, sequence_length, prediction_length)

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float()

# Create dataset and dataloader
train_dataset = data.TensorDataset(X_tensor, y_tensor)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the LSTM model
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs)

# After training, predict the next 4 time steps for a test sequence
model.eval()
test_seq = X_tensor[-1].unsqueeze(0)  # Take the last sequence in the dataset for testing
predicted_seq = model(test_seq).detach().numpy()
predicted_seq = scaler.inverse_transform(predicted_seq)

# Visualize CPU predictions (feature 0) against ground truth
plt.figure(figsize=(10, 6))
# Plot ground truth (last sequence used for testing, CPU is feature 0)
plt.plot(range(sequence_length), scaler.inverse_transform(X_tensor[-1])[:, 0], label='CPU (Ground Truth)')
# Plot predicted CPU usage for the next 4 time steps
plt.plot(range(sequence_length, sequence_length + prediction_length), predicted_seq[:, 0], label='CPU (Predicted)', linestyle='--')
plt.legend()
plt.title('CPU Prediction vs Ground Truth')
plt.xlabel('Time Steps')
plt.ylabel('CPU Usage')
plt.show()
