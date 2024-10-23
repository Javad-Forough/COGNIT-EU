import os
import numpy as np
from data import load_data, scale_data, create_sequences

def save_test_set():
    # Hyperparameters
    sequence_length = 99
    train_test_ratio = 0.8

    # Load and preprocess data
    filename = 'workload_series.npy'
    data_reshaped = load_data(filename)
    data_scaled, _ = scale_data(data_reshaped)  # No need for scaler in this case
    X, y = create_sequences(data_scaled, sequence_length, 1)

    # Split the data into training and testing sets
    train_size = int(len(X) * train_test_ratio)
    X_test, y_test = X[train_size:], y[train_size:]

    # Save the test set as .npy files
    save_dir = 'TestSet'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)

    print(f'Test set saved as .npy files in the {save_dir} directory.')

if __name__ == "__main__":
    save_test_set()
