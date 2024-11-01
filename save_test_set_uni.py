import os
import numpy as np
from data import load_data, scale_data, create_sequences

def save_test_set_univariate():
    # Hyperparameters
    sequence_length = 99
    train_test_ratio = 0.8

    # Load and preprocess data
    filename = 'workload_series.npy'
    data_reshaped = load_data(filename)
    data_scaled, _ = scale_data(data_reshaped)  # No need for scaler in this case

    # Define metrics/features for univariate prediction
    metrics = ['CPU', 'Memory', 'Disk Write', 'Network Received']

    # Create directory for saving test sets if it doesn't exist
    save_dir = 'TestSet_uni'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for feature_index, metric in enumerate(metrics):
        # Extract the feature column for univariate prediction
        y = data_scaled[:, feature_index]  # Univariate target for the current metric

        # Only take one feature column for X to ensure univariate structure
        data_scaled_univariate = data_scaled[:, feature_index:feature_index+1]

        # Create sequences for the current metric
        X, y = create_sequences(data_scaled_univariate, sequence_length, 1)

        # Split the data into training and testing sets
        train_size = int(len(X) * train_test_ratio)
        X_test, y_test = X[train_size:], y[train_size:]

        # Save the test set for the current metric as .npy files
        np.save(os.path.join(save_dir, f'X_test_{metric.lower()}.npy'), X_test)
        np.save(os.path.join(save_dir, f'y_test_{metric.lower()}.npy'), y_test)

        print(f'Test set saved for {metric} as .npy files in the {save_dir} directory.')

if __name__ == "__main__":
    save_test_set_univariate()
