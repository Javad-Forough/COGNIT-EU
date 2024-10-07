import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from data import load_data, scale_data, create_sequences
from myLSTM import LSTMModel, train_lstm_model  # Import LSTM
from myFFNN import FFNNModel, train_ffnn_model  # Import FFNN
import random

# Hyperparameters
model_type = 'FFNN'  # Change this to 'LSTM' to switch to LSTM
input_size = 4  
hidden_size = 64
num_layers = 2  # Only used for LSTM
output_size = 4  # Predicting the next 4 features
sequence_length = 99
prediction_length = 1
batch_size = 64
learning_rate = 0.001
num_epochs = 10
train_test_ratio = 0.8

# Load and preprocess data
print("Data preparation started.....")
filename = 'workload_series.npy'
data_reshaped = load_data(filename)
data_scaled, scaler = scale_data(data_reshaped)
X, y = create_sequences(data_scaled, sequence_length, prediction_length)

train_size = int(len(X) * train_test_ratio)

# Split the data into training and testing sets
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(len(X_test), len(y_test))

# Convert to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float()

# Create dataset and dataloader
train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

# Model selection and initialization
if model_type == 'LSTM':
    print("Initializing LSTM model...")
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the LSTM model
    print("Training LSTM model...")
    train_lstm_model(model, train_loader, criterion, optimizer, num_epochs)

    # Save the LSTM model
    model_save_path = 'lstm_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

elif model_type == 'FFNN':
    print("Initializing FFNN model...")
    model = FFNNModel(input_size * sequence_length, hidden_size, output_size)  # Input size adjusted for FFNN
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the FFNN model
    print("Training FFNN model...")
    train_ffnn_model(model, train_loader, criterion, optimizer, num_epochs)

    # Save the FFNN model
    model_save_path = 'ffnn_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')


# # Load the model's state dictionary
# model.load_state_dict(torch.load(model_save_path))

# Model evaluation starts here (same as before for both models)
# Randomly select 100 indices from the test set
X_test_tensor = torch.from_numpy(X_test).float()
random_indices = random.sample(range(len(X_test_tensor)), 100)

model.eval()
predictions = []
ground_truth = []

# Predict for 100 random samples
for idx in random_indices:
    with torch.no_grad():
        seq = X_test_tensor[idx].unsqueeze(0)
        
        if model_type == 'FFNN':
            seq = seq.view(seq.size(0), -1)  # Flatten input for FFNN

        pred = model(seq).detach().numpy()
        predictions.append(pred[0])  # Append predicted values
        gt = y_test[idx]
        ground_truth.append(gt)

# Rescale the predictions and ground truth back to original scale
predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, output_size))
ground_truth_rescaled = scaler.inverse_transform(np.array(ground_truth).reshape(-1, output_size))


# List of metric names for plotting
metric_names = ['CPU', 'Memory', 'Disk Write', 'Network Received']

# Colors for each metric (for better distinction)
colors = ['blue', 'green', 'red', 'purple']

# Line width for thicker lines
line_width = 2.5

# Iterate over all 4 metrics (features) and create separate plots
for i, metric_name in enumerate(metric_names):
    plt.figure(figsize=(12, 6))
    
    # Plot ground truth for all 100 random instances (feature `i`)
    plt.plot(range(len(ground_truth_rescaled)), ground_truth_rescaled[:, i], color='green', alpha=0.6, 
             label=f'{metric_name} (Ground Truth)', linewidth=line_width)
    
    # Plot predicted values for all 100 random test instances (feature `i`)
    plt.plot(range(len(predictions_rescaled)), predictions_rescaled[:, i], linestyle='--', color='purple', 
             alpha=0.6, label=f'{metric_name} (Predicted)', linewidth=line_width)
    
    plt.title(f'{metric_name} Predictions vs Ground Truth for 100 Random Test Samples')
    plt.xlabel('Time Steps')
    plt.ylabel(f'{metric_name} Usage')
    plt.grid()
    plt.legend([f'{metric_name} (Ground Truth)', f'{metric_name} (Predicted)'])
    
    # Save each plot as a separate file
    plt.savefig(f'{model_type.lower()}_{metric_name}_predictions_vs_ground_truth.png')
    
    # Show the plot
    plt.show()
