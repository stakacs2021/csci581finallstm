# Quick Start Guide - LSTM Stock Price Prediction

## What Was Implemented

A minimal working LSTM model that fulfills all project requirements:

✅ **TensorFlow/Keras LSTM** - Top-down approach  
✅ **Time Series Split** - 70% train, 15% validation, 15% test  
✅ **Data Normalization** - Min-Max scaling  
✅ **Adam Optimizer + MSE Loss** - Gradient descent with backpropagation  
✅ **RMSE and MAE Metrics** - Calculated on test set  
✅ **GPU-Ready** - Configured for A100 with mixed precision  
✅ **Visualizations** - True vs predicted prices, training history, error analysis  
✅ **System Specs** - CPU cores, GPU info, memory  

## Files Created

- `train_lstm.py` - Main training script (ready to run)
- `requirements.txt` - Updated with scikit-learn and psutil

## How to Run

### 1. Install Dependencies
```bash
# Activate your virtual environment
source 581lstm/bin/activate

# Install/update requirements
pip install -r requirements.txt
```

### 2. Run Training
```bash
python train_lstm.py
```

### 3. What Happens

The script will:
1. Download 5 years of AAPL stock data from Yahoo Finance
2. Normalize the data (Min-Max scaling)
3. Create sequences (20 days → predict next day)
4. Split into train/val/test (70/15/15)
5. Build and train LSTM model
6. Evaluate on test set (RMSE, MAE)
7. Generate visualizations:
   - `predictions_vs_actual.png` - True vs predicted prices
   - `training_history.png` - Loss and MAE curves
   - `prediction_errors.png` - Error analysis
8. Display system specifications

### 4. Output Files

- `best_lstm_model.keras` - Saved best model (based on validation loss)
- `predictions_vs_actual.png` - Main visualization
- `training_history.png` - Training curves
- `prediction_errors.png` - Error analysis

## Model Architecture

```
Input(20 timesteps, 1 feature)
  ↓
LSTM(64 units)
  ↓
Dropout(0.2)
  ↓
Dense(32, relu)
  ↓
Dense(1) → Next day closing price
```

## Key Features

- **Early Stopping**: Stops if validation loss doesn't improve for 10 epochs
- **Learning Rate Reduction**: Reduces LR by 50% if validation loss plateaus
- **Model Checkpointing**: Saves best model automatically
- **Mixed Precision**: Uses float16 for faster training on A100
- **GPU Memory Growth**: Prevents GPU memory allocation issues

## Customization

You can easily modify:
- **Ticker**: Change `ticker = "AAPL"` to any stock symbol
- **Window Size**: Change `window_size = 20` (days to look back)
- **Epochs**: Change `epochs=50` in the fit() call
- **Batch Size**: Change `batch_size=32`
- **LSTM Units**: Change `layers.LSTM(64, ...)` to 32, 128, etc.

## Next Steps

Once this basic version works:
1. Experiment with different window sizes
2. Add more features (volume, technical indicators)
3. Try different architectures (stacked LSTM, bidirectional)
4. Tune hyperparameters
5. Test on multiple stocks

## Troubleshooting

**GPU not found?**
- Script will automatically use CPU
- Check GPU with: `nvidia-smi`

**Out of memory?**
- Reduce batch_size (try 16 or 8)
- Reduce LSTM units (try 32 instead of 64)

**Poor predictions?**
- Stock prices are hard to predict - this is expected
- Try different window sizes
- Add more features
- Train for more epochs

