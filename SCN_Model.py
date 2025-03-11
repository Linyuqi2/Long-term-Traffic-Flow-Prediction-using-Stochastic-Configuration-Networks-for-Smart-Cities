import numpy as np
import pandas as pd
from SCN import SCN
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

def smape(A, F):
    """Calculate SMAPE (Symmetric Mean Absolute Percentage Error)。"""
    return 100 / len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

# File path and data column name
filepath = 'guangzhou_r1.csv'

column_name = 'speed'
step = 12
step_out = 1

# Initialize the parameters of the SCN model
L_max = 100  # Maximum number of hidden nodes
tol = 0.0001
T_max = 100  # Maximum number of candidate nodes
Lambdas = [0.001, 0.005, 0.008, 0.01, 0.1, 0.5, 1, 5, 10, 30, 50, 100, 150, 200, 250]
r = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999]
nB = 1
verbose = 10

# Create an SCN model instance
M = SCN(L_max, T_max, tol, Lambdas, r, nB, verbose)

# Load data and prepare data functions
def load_csv_data(filepath, column_name):
    """Load file from CSV"""
    data = pd.read_csv(filepath)
    return data[column_name].values

def prepare_data(data, n_steps_in, n_steps_out):
    """ Prepares data for model training and testing according to the specified number of steps."""
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        seq_x = data[i:(i + n_steps_in)]
        seq_y = data[(i + n_steps_in):(i + n_steps_in + n_steps_out)]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Prepare data
data = load_csv_data(filepath, column_name)
n_steps_in, n_steps_out = step, step_out
X, y = prepare_data(data, n_steps_in, n_steps_out)

# Split data into training set and test set
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train the model using a training set
per = M.regression(X_train, y_train)

# Use test sets to make predictions
predictions = np.array([M.getOutput(X_test[i].reshape(1, n_steps_in)) for i in range(len(X_test))])
print(predictions.shape)

# Adjust the shape of the forecast results
if len(predictions.shape) == 3 and predictions.shape[1] == 1:
    predictions = predictions.squeeze(axis=1)

# Calculate and output performance metrics
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
smape_value = smape(y_test.flatten(), predictions.flatten())
r2 = r2_score(y_test, predictions)

# Print out performance matrics
metrics = f"RMSE: {rmse}, MAE: {mae}, SMAPE: {smape_value}, R2: {r2}"
print("测试集上的性能指标:", metrics)

plt.figure(figsize=(12, 6))
plt.plot(y_test.flatten(), label='Actual Values', color='blue')
plt.plot(predictions.flatten(), label='Predictions', color='red', linestyle='--')
plt.title('Comparison of Actual and Predicted Values')
plt.xlabel('Time Steps')
plt.ylabel('Longitude')
plt.legend()
plt.grid(True)
plt.show()



# 绘图

