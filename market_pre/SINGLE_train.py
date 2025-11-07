# SINGLE_LSTM_TRAIN.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducible results
np.random.seed(42)
tf.random.set_seed(42)

# Load preprocessed data
print("Loading preprocessed data...")
exec(open('/home/holo/market_pre/data_process.py').read())

# Select single asset for training (using AMZN as example)
asset_name = 'DPZ'  # Can be changed to 'DPZ', 'BTC', 'NFLX'

# Use sequential data for LSTM training, but select only one asset
X_train_seq_full = processed_data['X_seq_train']
X_test_seq_full = processed_data['X_seq_test']
y_train_seq_full = processed_data['y_seq_train']
y_test_seq_full = processed_data['y_seq_test']
scaler_y_full = processed_data['scaler_y']

# Select data for single asset (AMZN is column 0)
y_train_seq_single = y_train_seq_full[:, 0:1]  # Keep 2D array
y_test_seq_single = y_test_seq_full[:, 0:1]
scaler_y_single = MinMaxScaler()

# Rescale single asset data
y_train_single_prices = processed_data['y_train'][:, 0:1]
y_test_single_prices = processed_data['y_test'][:, 0:1]
y_train_single_scaled = scaler_y_single.fit_transform(y_train_single_prices)
y_test_single_scaled = scaler_y_single.transform(y_test_single_prices)

# Create sequences for single asset data
def create_single_sequences(data, seq_length=10):
    """Create time series data for single asset"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Recreate sequence data using only single asset historical prices
seq_length = 10
single_asset_prices = processed_data['df_processed'][asset_name].values.reshape(-1, 1)
single_asset_scaled = scaler_y_single.fit_transform(single_asset_prices)

X_seq_single, y_seq_single = create_single_sequences(single_asset_scaled, seq_length)

# Split into train/test sets
split_idx = int(len(X_seq_single) * 0.8)
X_train_seq_single = X_seq_single[:split_idx]
X_test_seq_single = X_seq_single[split_idx:]
y_train_seq_single = y_seq_single[:split_idx]
y_test_seq_single = y_seq_single[split_idx:]

print(f"Single asset ({asset_name}) training data shape: {X_train_seq_single.shape}")
print(f"Single asset ({asset_name}) testing data shape: {X_test_seq_single.shape}")

# Build LSTM model
def create_single_lstm_model(input_shape):
    """
    Create LSTM model for single asset
    """
    model = Sequential()
    
    # Add LSTM layers
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Add dense layers
    model.add(Dense(units=25, activation='relu'))
    model.add(Dense(units=1))  # Predict single asset price
    
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='mse', metrics=['mae'])
    
    return model

# Create model
input_shape = (X_train_seq_single.shape[1], X_train_seq_single.shape[2])
model = create_single_lstm_model(input_shape)

print("Model structure:")
model.summary()

# Train model
print("Starting model training...")
history = model.fit(
    X_train_seq_single, y_train_seq_single,
    epochs=500,
    batch_size=32,
    validation_data=(X_test_seq_single, y_test_seq_single),
    verbose=1,
    shuffle=False
)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Model prediction
print("Making predictions...")
train_predictions = model.predict(X_train_seq_single)
test_predictions = model.predict(X_test_seq_single)

# Inverse transform predictions
train_predictions_rescaled = scaler_y_single.inverse_transform(train_predictions)
test_predictions_rescaled = scaler_y_single.inverse_transform(test_predictions)
y_train_actual = scaler_y_single.inverse_transform(y_train_seq_single)
y_test_actual = scaler_y_single.inverse_transform(y_test_seq_single)

# Calculate evaluation metrics
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions_rescaled))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions_rescaled))
test_mae = mean_absolute_error(y_test_actual, test_predictions_rescaled)

print(f"\nModel Evaluation Results ({asset_name}):")
print(f"Training RMSE: {train_rmse:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}")
print(f"Testing MAE: {test_mae:.2f}")

# Plot predictions
plt.figure(figsize=(12, 6))

# Plot actual values
plt.plot(y_test_actual, label=f'Actual {asset_name} Price', color='blue')

# Plot predicted values
plt.plot(test_predictions_rescaled, label=f'Predicted {asset_name} Price', color='red', linestyle='--')

plt.title(f'{asset_name} Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Predict future prices
def predict_future_single(model, last_sequence, scaler, days=5):
    """
    Predict future prices
    
    Parameters:
    model: Trained LSTM model
    last_sequence: Last sequence data
    scaler: Scaler for inverse transformation
    days: Number of days to predict
    
    Returns:
    Future days predicted prices
    """
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        # Predict next day
        next_pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))
        future_predictions.append(next_pred[0])
        
        # Update sequence: remove oldest day, add predicted day
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = next_pred[0]
    
    # Inverse transform predictions
    future_predictions = np.array(future_predictions)
    future_predictions_rescaled = scaler.inverse_transform(future_predictions)
    
    return future_predictions_rescaled

# Use last 10 days data to predict next 5 days
last_sequence = X_test_seq_single[-1]
future_prices = predict_future_single(model, last_sequence, scaler_y_single, days=5)

print(f"\nFuture 5 Days {asset_name} Price Prediction:")
print(future_prices.flatten())

# Save model
model.save(f'/home/holo/market_pre/lstm_{asset_name.lower()}_model.keras')
print(f"\nModel saved to: /home/holo/market_pre/lstm_{asset_name.lower()}_model.keras")

print(f"\n{asset_name} model training and prediction completed!")