"""
Minimal LSTM Stock Price Prediction Model
Fulfills project requirements:
- TensorFlow/Keras LSTM
- Train/Val/Test split (time series)
- RMSE and MAE metrics
- Data normalization
- Adam optimizer, MSE loss
- GPU-ready training
- Prediction visualization
"""

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# GPU Configuration for A100
print("=" * 60)
print("GPU Configuration")
print("=" * 60)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found, using CPU")

# Enable mixed precision for faster training on A100
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("Mixed precision enabled for faster training")
print()

# ============================================================================
# 1. DATA COLLECTION
# ============================================================================
print("=" * 60)
print("Step 1: Downloading Stock Data")
print("=" * 60)
ticker = "AAPL"
print(f"Downloading {ticker} stock data from Yahoo Finance...")
df = yf.download(ticker, period="5y", interval="1d")

# Use only closing price
df = df[["Close"]].dropna()
print(f"Downloaded {len(df)} days of data")
print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Sample data:\n{df.head()}")
print()

# ============================================================================
# 2. DATA PREPROCESSING - Normalization
# ============================================================================
print("=" * 60)
print("Step 2: Data Normalization")
print("=" * 60)
values = df["Close"].values.reshape(-1, 1).astype("float32")

# Min-Max scaling: (x - min) / (max - min)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values).flatten()

print(f"Original price range: ${values.min():.2f} - ${values.max():.2f}")
print(f"Scaled value range: {scaled.min():.4f} - {scaled.max():.4f}")
print()

# ============================================================================
# 3. CREATE SEQUENCES
# ============================================================================
print("=" * 60)
print("Step 3: Creating Sequences")
print("=" * 60)
window_size = 20  # Look back 20 days to predict next day

def create_dataset(series, window):
    """Create sequences for LSTM: X = past window_size days, y = next day"""
    X_list = []
    y_list = []
    for i in range(len(series) - window):
        X_list.append(series[i : i + window])
        y_list.append(series[i + window])
    return np.array(X_list), np.array(y_list)

X, y = create_dataset(scaled, window_size)

# Reshape for LSTM: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

print(f"Dataset shapes:")
print(f"  X: {X.shape} (samples, timesteps, features)")
print(f"  y: {y.shape} (samples,)")
print()

# ============================================================================
# 4. TIME SERIES SPLIT (Train/Val/Test)
# ============================================================================
print("=" * 60)
print("Step 4: Time Series Split")
print("=" * 60)
# Maintain temporal order: 70% train, 15% val, 15% test
n_samples = len(X)
train_end = int(n_samples * 0.70)
val_end = int(n_samples * 0.85)

X_train = X[:train_end]
y_train = y[:train_end]
X_val = X[train_end:val_end]
y_val = y[train_end:val_end]
X_test = X[val_end:]
y_test = y[val_end:]

print(f"Train set: {len(X_train)} samples ({len(X_train)/n_samples*100:.1f}%)")
print(f"Val set:   {len(X_val)} samples ({len(X_val)/n_samples*100:.1f}%)")
print(f"Test set:  {len(X_test)} samples ({len(X_test)/n_samples*100:.1f}%)")
print()

# ============================================================================
# 5. BUILD LSTM MODEL
# ============================================================================
print("=" * 60)
print("Step 5: Building LSTM Model")
print("=" * 60)
model = models.Sequential([
    layers.Input(shape=(window_size, 1)),
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output: next day closing price (scaled)
])

# Compile with Adam optimizer and MSE loss
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]  # Mean Absolute Error as additional metric
)

print("Model architecture:")
model.summary()
print()

# ============================================================================
# 6. TRAINING
# ============================================================================
print("=" * 60)
print("Step 6: Training Model")
print("=" * 60)

# Callbacks for training
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_lstm_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,  # Will stop early if validation loss doesn't improve
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining complete!")
print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
print(f"Best validation MAE: {min(history.history['val_mae']):.6f}")
print()

# ============================================================================
# 7. EVALUATION - RMSE and MAE on Test Set
# ============================================================================
print("=" * 60)
print("Step 7: Evaluation on Test Set")
print("=" * 60)

# Load best model
model.load_weights('best_lstm_model.keras')

# Make predictions on test set
y_test_pred_scaled = model.predict(X_test, verbose=0).flatten()

# Inverse transform to get actual prices
y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_test_pred = scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
mae = mean_absolute_error(y_test_true, y_test_pred)

print(f"Test Set Results:")
print(f"  RMSE: ${rmse:.2f}")
print(f"  MAE:  ${mae:.2f}")
print(f"  RMSE as % of avg price: {(rmse / y_test_true.mean()) * 100:.2f}%")
print()

# ============================================================================
# 8. VISUALIZATION - True vs Predicted Prices
# ============================================================================
print("=" * 60)
print("Step 8: Creating Visualizations")
print("=" * 60)

# Get dates for test set
test_dates = df.index[val_end + window_size:]

# Plot 1: True vs Predicted Closing Prices
plt.figure(figsize=(14, 6))
plt.plot(test_dates, y_test_true, label='Actual Closing Price', linewidth=2, alpha=0.8)
plt.plot(test_dates, y_test_pred, label='Predicted Closing Price', linewidth=2, alpha=0.8)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Closing Price ($)', fontsize=12)
plt.title(f'{ticker} Stock Price Prediction - LSTM Model', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('predictions_vs_actual.png', dpi=150, bbox_inches='tight')
print("Saved: predictions_vs_actual.png")
plt.show()

# Plot 2: Training History
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('MSE Loss', fontsize=11)
plt.title('Model Loss', fontsize=12, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='Val MAE', linewidth=2)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('MAE', fontsize=11)
plt.title('Mean Absolute Error', fontsize=12, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("Saved: training_history.png")
plt.show()

# Plot 3: Prediction Errors
errors = y_test_true - y_test_pred
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(test_dates, errors, linewidth=1.5, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Date', fontsize=11)
plt.ylabel('Prediction Error ($)', fontsize=11)
plt.title('Prediction Errors Over Time', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Prediction Error ($)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Error Distribution', fontsize=12, fontweight='bold')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('prediction_errors.png', dpi=150, bbox_inches='tight')
print("Saved: prediction_errors.png")
plt.show()

print("\n" + "=" * 60)
print("All visualizations saved!")
print("=" * 60)

# ============================================================================
# 9. SYSTEM SPECIFICATIONS (for project requirement)
# ============================================================================
print("\n" + "=" * 60)
print("System Specifications")
print("=" * 60)
print(f"CPU Cores: {os.cpu_count()}")
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"GPU {i}: {gpu.name}")
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            print(f"  Device Details: {gpu_details}")
        except:
            pass
else:
    print("GPU: None (CPU only)")

# Memory info
import psutil
memory = psutil.virtual_memory()
print(f"Total Memory: {memory.total / (1024**3):.2f} GB")
print(f"Available Memory: {memory.available / (1024**3):.2f} GB")
print()

print("=" * 60)
print("Training and Evaluation Complete!")
print("=" * 60)

