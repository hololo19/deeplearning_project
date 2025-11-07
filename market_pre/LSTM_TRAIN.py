# LSTM_TRAIN.py
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

# Use sequential data for LSTM training
X_train_seq = processed_data['X_seq_train']
X_test_seq = processed_data['X_seq_test']
y_train_seq = processed_data['y_seq_train']
y_test_seq = processed_data['y_seq_test']
scaler_y = processed_data['scaler_y']

print(f"Training data shape: {X_train_seq.shape}")
print(f"Testing data shape: {X_test_seq.shape}")

# Build LSTM model
def create_lstm_model(input_shape):
    """
    Create LSTM model
    """
    model = Sequential()
    
    model = Sequential()
    
    # 增加更多层和神经元
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # 添加全连接层
    model.add(Dense(units=25, activation='relu'))
    model.add(Dense(units=4))  # 预测4个资产价格
    
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='mse', metrics=['mae'])
    
    return model

# Create model
input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
model = create_lstm_model(input_shape)

print("Model structure:")
model.summary()

# Train model
print("Start training model...")
history = model.fit(
    X_train_seq, y_train_seq,
    epochs=500,
    batch_size=64,
    validation_data=(X_test_seq, y_test_seq),
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
train_predictions = model.predict(X_train_seq)
test_predictions = model.predict(X_test_seq)

# Inverse transform predictions
train_predictions_rescaled = scaler_y.inverse_transform(train_predictions)
test_predictions_rescaled = scaler_y.inverse_transform(test_predictions)
y_train_actual = scaler_y.inverse_transform(y_train_seq)
y_test_actual = scaler_y.inverse_transform(y_test_seq)

# Calculate evaluation metrics
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions_rescaled))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions_rescaled))
test_mae = mean_absolute_error(y_test_actual, test_predictions_rescaled)

print(f"\nModel Evaluation Results:")
print(f"Training RMSE: {train_rmse:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}")
print(f"Testing MAE: {test_mae:.2f}")

# Plot predictions
assets = ['AMZN', 'DPZ', 'BTC', 'NFLX']
plt.figure(figsize=(15, 10))

for i, asset in enumerate(assets):
    plt.subplot(2, 2, i+1)
    
    # Plot actual values
    plt.plot(y_test_actual[:, i], label=f'Actual {asset} Price', color='blue')
    
    # Plot predicted values
    plt.plot(test_predictions_rescaled[:, i], label=f'Predicted {asset} Price', color='red', linestyle='--')
    
    plt.title(f'{asset} Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Predict future prices
def predict_future(model, last_sequence, scaler, days=5):
    """
    Predict future prices
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
last_sequence = X_test_seq[-1]
future_prices = predict_future(model, last_sequence, scaler_y, days=5)

print(f"\nFuture 5 Days Price Prediction:")
for i, asset in enumerate(assets):
    print(f"{asset}: {future_prices[:, i]}")

# Save model
model.save('/home/holo/market_pre/lstm_stock_model.keras')
print("\nModel saved to: /home/holo/market_pre/lstm_stock_model.keras")

# Create prediction function for future use
def predict_next_days(data, model, scaler, days=5):
    """
    Predict future prices using latest data
    
    Parameters:
    data: Latest normalized data sequence
    model: Trained LSTM model
    scaler: Scaler for inverse transformation
    days: Number of days to predict
    
    Returns:
    Future days predicted prices
    """
    # Get last seq_length days data
    last_sequence = data[-10:]  # Assuming seq_length=10
    
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        # Predict next day
        next_pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))
        future_predictions.append(next_pred[0])
        
        # Update sequence
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = next_pred[0]
    
    # Inverse transform
    future_predictions = np.array(future_predictions)
    future_predictions_rescaled = scaler.inverse_transform(future_predictions)
    
    return future_predictions_rescaled

print("\nModel training and prediction completed!")
print("You can use predict_next_days() function to predict future stock prices")