# Demo code for final project (building LSTM & Loading data from YF)

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras import layers, models

#Downloading the data from YF 
print("Downloading AAPL stock data from Yahoo Finance...")
df = yf.download("AAPL", period="5y", interval="1d")

# demo code for closing price
df = df[["Close"]].dropna()

print("Sample of downloaded data:")
print(df.head())

#  prep window size for LSTM
window_size = 20

values = df["Close"].values.astype("float32")


def create_dataset(series, window):
    X_list = []
    y_list = []
    for i in range(len(series) - window):
        X_list.append(series[i : i + window])
        y_list.append(series[i + window])
    return np.array(X_list), np.array(y_list)


X, y = create_dataset(scaled, window_size)

# reshape, with just close price as feature
X = X.reshape((X.shape[0], X.shape[1], 1))

print("Dataset shapes:")
print("X:", X.shape, "(samples, timesteps, features)")
print("y:", y.shape, "(samples,)")

# split into train and test
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

#build mini model
model = models.Sequential(
    [
        layers.Input(shape=(window_size, 1)),
        layers.LSTM(32),
        layers.Dense(1),
    ]
)

model.compile(optimizer="adam", loss="mse")

print("Model summary:")
model.summary()

#sample training for 1 epoch
print("Starting training for 1 epoch (demo)...")
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=1,
    batch_size=32,
    verbose=1,
)

print("Training complete. Example loss values:")
print("Train loss:", history.history["loss"][-1])
print("Val loss:  ", history.history["val_loss"][-1])