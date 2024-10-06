import numpy as np
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
print("Data preparation started.....")
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
print("Model initialization started.....")
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
print("Model training started.....")
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Save the model
model_save_path = 'lstm_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

# # Create a new instance of the model
# model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# # Load the model's state dictionary
# model.load_state_dict(torch.load('lstm_model.pth'))

# After training, predict the next 4 time steps for a test sequence
print("Model evaluation started.....")
# Split data into training and testing sets
train_size = int(len(data_scaled) * 0.8)
train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]

# Create sequences from the test data
def create_test_sequences(data, seq_length):
    test_sequences = []
    for i in range(len(data) - seq_length + 1):
        test_sequences.append(data[i:i + seq_length])
    return np.array(test_sequences)

# Create test sequences
test_sequences = create_test_sequences(test_data, sequence_length)

# Convert to PyTorch tensors
test_sequences_tensor = torch.from_numpy(test_sequences).float()

# Evaluate the model
model.eval()
predictions = []
ground_truth = []

# Use a sliding window to generate predictions
for i in range(len(test_sequences_tensor)):
    with torch.no_grad():
        seq = test_sequences_tensor[i].unsqueeze(0)  # Add batch dimension
        pred = model(seq).detach().numpy()
        predictions.append(pred[0])  # Append predicted values
        # Ground truth is the next 4 time steps after the sequence
        if i + prediction_length <= len(test_data):
            gt = test_data[i + sequence_length:i + sequence_length + prediction_length]
            ground_truth.append(gt)

# Convert to numpy arrays for easier indexing
predictions = np.array(predictions)
ground_truth = np.array(ground_truth)

# Visualize CPU predictions (feature 0) against ground truth for the entire test set
plt.figure(figsize=(12, 6))

# Plot ground truth for all instances (CPU is feature 0)
for i in range(len(ground_truth)):
    plt.plot(range(i * prediction_length, (i + 1) * prediction_length), scaler.inverse_transform(ground_truth[i])[:, 0], color='blue', alpha=0.3)

# Plot predicted CPU usage for the next 4 time steps
for i in range(len(predictions)):
    plt.plot(range(i * prediction_length, (i + 1) * prediction_length), scaler.inverse_transform(predictions[i].reshape(1, -1))[:, 0], linestyle='--', color='orange', alpha=0.3)

plt.title('CPU Predictions vs Ground Truth for Test Set')
plt.xlabel('Time Steps')
plt.ylabel('CPU Usage')
plt.grid()
plt.legend(['CPU (Ground Truth)', 'CPU (Predicted)'])
plt.show()
