from sklearn.metrics import mean_squared_error

def calculate_rmse(predictions, ground_truth):
    return np.sqrt(mean_squared_error(ground_truth, predictions))