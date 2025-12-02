# Implementation and Testing Plan
## Predicting Daily Stock Closing Prices Using LSTM Recurrent Neural Network

---

## 1. Project Overview

**Objective**: Build and evaluate an LSTM model to predict daily stock closing prices using historical data from Yahoo Finance.

**Key Requirements**:
- Use TensorFlow/Keras (top-down approach)
- Time series train/validation/test split
- Evaluate with RMSE and MAE
- Use A100 GPU on cscigpu for training
- Handle noisy, non-stationary data
- Feature engineering and selection
- Visualize predictions vs actual prices

---

## 2. Implementation Phases

### Phase 1: Data Collection and Preprocessing (Week 1)

#### 1.1 Data Collection
- **Tool**: `yfinance` library
- **Stocks**: Start with AAPL, expand to multiple stocks if time permits
- **Time Period**: 5-10 years of daily data
- **Features to Collect**:
  - Open, High, Low, Close, Volume (OHLCV)
  - Additional indicators (calculated):
    - Moving averages (SMA, EMA)
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - Volume indicators

#### 1.2 Data Preprocessing
- **Handle Missing Data**:
  - Forward fill for minor gaps
  - Remove rows with significant missing data
- **Data Normalization**:
  - Min-Max scaling: `(x - min) / (max - min)` per feature
  - Alternative: Standardization `(x - mean) / std`
  - Store scaler parameters for inverse transformation
- **Stationarity**:
  - Apply differencing if needed: `price_diff = price[t] - price[t-1]`
  - Log returns: `log(price[t] / price[t-1])`
  - Test for stationarity using Augmented Dickey-Fuller test

#### 1.3 Feature Engineering
- **Technical Indicators** (using `pandas_ta` or manual calculation):
  - Simple Moving Average (SMA) - 7, 14, 30 day windows
  - Exponential Moving Average (EMA) - 7, 14, 30 day windows
  - RSI (14-day period)
  - MACD (12, 26, 9)
  - Bollinger Bands (20-day, 2 std)
  - Volume-based features (volume moving averages)
- **Time-based Features**:
  - Day of week, month, quarter
  - Market regime indicators
- **Target Variable**:
  - Next day closing price (regression)
  - Or: Price change percentage (classification alternative)

#### 1.4 Data Validation
- Check for data quality:
  - No negative prices
  - Volume >= 0
  - High >= Low
  - Close within [Low, High]
- Visual inspection of data plots
- Statistical summary (mean, std, min, max, skewness)

---

### Phase 2: Model Architecture Design (Week 1-2)

#### 2.1 LSTM Architecture Options

**Option A: Simple LSTM**
```python
Input(shape=(window_size, n_features))
LSTM(64, return_sequences=True)
Dropout(0.2)
LSTM(32, return_sequences=False)
Dropout(0.2)
Dense(16, activation='relu')
Dense(1)  # Output: next day closing price
```

**Option B: Stacked LSTM with Attention**
```python
Input(shape=(window_size, n_features))
LSTM(128, return_sequences=True)
LSTM(64, return_sequences=True)
Attention()  # Optional attention mechanism
LSTM(32, return_sequences=False)
Dense(1)
```

**Option C: Bidirectional LSTM**
```python
Input(shape=(window_size, n_features))
Bidirectional(LSTM(64, return_sequences=True))
Bidirectional(LSTM(32, return_sequences=False))
Dense(1)
```

#### 2.2 Hyperparameters to Tune
- **Window Size**: 10, 20, 30, 60 days (sliding window)
- **LSTM Units**: 32, 64, 128, 256
- **Number of Layers**: 1, 2, 3
- **Dropout Rate**: 0.1, 0.2, 0.3
- **Batch Size**: 16, 32, 64, 128
- **Learning Rate**: 0.001, 0.0001, 0.00001
- **Optimizer**: Adam, RMSprop, SGD with momentum

#### 2.3 Model Compilation
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (with learning rate scheduling)
- **Metrics**: MAE, RMSE (custom metrics)
- **Callbacks**:
  - EarlyStopping (patience=10, monitor='val_loss')
  - ReduceLROnPlateau (factor=0.5, patience=5)
  - ModelCheckpoint (save best model)
  - TensorBoard (for visualization)

---

### Phase 3: Data Splitting and Sequence Creation (Week 2)

#### 3.1 Time Series Split Strategy
- **Train Set**: 70% (chronologically first)
- **Validation Set**: 15% (middle)
- **Test Set**: 15% (most recent, future data)
- **Important**: No shuffling - maintain temporal order

#### 3.2 Sequence Creation Function
```python
def create_sequences(data, window_size, forecast_horizon=1):
    """
    Create sequences for LSTM input
    - data: normalized feature matrix
    - window_size: number of timesteps to look back
    - forecast_horizon: number of days ahead to predict (1 for next day)
    Returns: X (samples, timesteps, features), y (samples,)
    """
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size+forecast_horizon-1, target_idx])
    return np.array(X), np.array(y)
```

#### 3.3 Data Shape Verification
- X shape: `(n_samples, window_size, n_features)`
- y shape: `(n_samples,)`
- Ensure no data leakage between splits

---

### Phase 4: Model Training (Week 2-3)

#### 4.1 Training Setup for A100 GPU
- **Environment**: cscigpu cluster with A100
- **GPU Configuration**:
  ```python
  # Enable mixed precision for faster training
  tf.keras.mixed_precision.set_global_policy('mixed_float16')
  
  # GPU memory growth
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
      tf.config.experimental.set_memory_growth(gpus[0], True)
  ```

#### 4.2 Training Strategy
- **Mini-batch Training**: Use batch sizes that fit GPU memory (32-128)
- **Epochs**: Start with 50-100, use early stopping
- **Validation**: Monitor validation loss after each epoch
- **Parallelism**: 
  - Data parallelism (if multiple GPUs)
  - Batch processing for faster I/O

#### 4.3 Training Loop
```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, lr_scheduler, checkpoint, tensorboard],
    verbose=1
)
```

#### 4.4 Hyperparameter Tuning
- Use `keras-tuner` or manual grid search
- Focus on: window_size, LSTM units, dropout, learning rate
- Train multiple models and compare validation performance

---

### Phase 5: Model Evaluation (Week 3)

#### 5.1 Evaluation Metrics

**Primary Metrics**:
- **RMSE** (Root Mean Squared Error):
  ```python
  rmse = np.sqrt(mean_squared_error(y_true, y_pred))
  ```
- **MAE** (Mean Absolute Error):
  ```python
  mae = mean_absolute_error(y_true, y_pred)
  ```

**Additional Metrics**:
- **MAPE** (Mean Absolute Percentage Error)
- **Directional Accuracy**: % of correct up/down predictions
- **R² Score**: Coefficient of determination

#### 5.2 Evaluation on Test Set
- **Important**: Only evaluate on test set once, after final model selection
- Calculate metrics on:
  - Train set (check for overfitting)
  - Validation set (model selection)
  - Test set (final evaluation)

#### 5.3 Visualization
- **Plot 1**: True vs Predicted prices over time
  ```python
  plt.plot(test_dates, y_test_true, label='Actual', linewidth=2)
  plt.plot(test_dates, y_test_pred, label='Predicted', linewidth=2)
  plt.xlabel('Date')
  plt.ylabel('Closing Price')
  plt.title('LSTM Stock Price Predictions')
  plt.legend()
  plt.show()
  ```

- **Plot 2**: Prediction error distribution (histogram)
- **Plot 3**: Residuals plot (errors over time)
- **Plot 4**: Scatter plot: Predicted vs Actual

#### 5.4 Error Analysis
- Identify periods with high prediction error
- Analyze if errors correlate with:
  - Market volatility
  - Economic events
  - Specific stock characteristics
- Calculate error statistics by:
  - Day of week
  - Month/quarter
  - Market regime (bull/bear)

---

### Phase 6: Model Optimization and Refinement (Week 3-4)

#### 6.1 Addressing Challenges

**Challenge 1: Noisy, Non-stationary Data**
- Solutions:
  - Apply differencing or log returns
  - Use rolling window normalization
  - Add volatility features (rolling std)
  - Consider regime detection

**Challenge 2: Feature Selection**
- Solutions:
  - Feature importance analysis (permutation importance)
  - Correlation analysis
  - Recursive feature elimination
  - Compare models with different feature sets

**Challenge 3: LSTM Architecture**
- Solutions:
  - Experiment with different architectures
  - Add regularization (dropout, L2)
  - Try attention mechanisms
  - Ensemble multiple models

**Challenge 4: Parallelism and Correctness**
- Solutions:
  - Verify GPU utilization (nvidia-smi)
  - Use TensorFlow profiler
  - Ensure reproducible results (set random seeds)
  - Validate data pipeline correctness

#### 6.2 Model Improvements
- **Ensemble Methods**: Average predictions from multiple models
- **Transfer Learning**: Pre-train on multiple stocks, fine-tune on target
- **Multi-task Learning**: Predict price and volatility simultaneously
- **Online Learning**: Update model with new data periodically

---

## 3. Testing Plan

### 3.1 Unit Tests

**Data Preprocessing Tests**:
- Test data normalization (min-max, standardization)
- Test sequence creation function
- Test train/val/test split maintains temporal order
- Test handling of missing data

**Model Tests**:
- Test model architecture compiles
- Test forward pass produces correct output shape
- Test loss calculation
- Test gradient computation

### 3.2 Integration Tests

- Test full pipeline: data → preprocessing → model → prediction
- Test model saving and loading
- Test prediction on new data
- Test GPU utilization during training

### 3.3 Validation Tests

- **Overfitting Check**: Compare train vs validation loss
- **Generalization**: Test on different stocks
- **Temporal Validation**: Test on different time periods
- **Robustness**: Test with noisy/missing data

### 3.4 Performance Tests

- **Training Speed**: Measure time per epoch on A100
- **Inference Speed**: Measure prediction time
- **Memory Usage**: Monitor GPU/CPU memory
- **Scalability**: Test with different data sizes

---

## 4. Deliverables Checklist

### Code Files
- [ ] `data_collection.py`: Download and save stock data
- [ ] `data_preprocessing.py`: Normalization, feature engineering
- [ ] `model_architecture.py`: LSTM model definitions
- [ ] `train.py`: Training script with GPU support
- [ ] `evaluate.py`: Evaluation metrics and visualization
- [ ] `predict.py`: Make predictions on new data
- [ ] `utils.py`: Helper functions
- [ ] `config.py`: Hyperparameters and configuration

### Documentation
- [ ] README.md: Project overview and setup instructions
- [ ] IMPLEMENTATION_PLAN.md: This document
- [ ] Model architecture diagram
- [ ] Results report with metrics and visualizations

### Results
- [ ] Trained model file (.h5 or .keras)
- [ ] Training history plots (loss curves)
- [ ] Evaluation metrics (RMSE, MAE) on test set
- [ ] Prediction vs actual price plots
- [ ] System specifications output (cores, GPU, memory)

---

## 5. Timeline and Milestones

### Week 1
- **Milestone 1**: Data collection and preprocessing complete
- **Milestone 2**: Feature engineering implemented
- **Deliverable**: Clean, normalized dataset with features

### Week 2
- **Milestone 3**: LSTM architecture designed and implemented
- **Milestone 4**: Training pipeline set up on A100
- **Deliverable**: Working training script with GPU support

### Week 3
- **Milestone 5**: Model trained and hyperparameters tuned
- **Milestone 6**: Evaluation metrics implemented
- **Deliverable**: Trained model with validation results

### Week 4
- **Milestone 7**: Final evaluation on test set
- **Milestone 8**: Visualizations and analysis complete
- **Deliverable**: Final report with results and visualizations

---

## 6. Risk Mitigation

### Risk 1: Data Quality Issues
- **Mitigation**: Implement robust data validation, use multiple data sources

### Risk 2: Model Overfitting
- **Mitigation**: Use dropout, early stopping, cross-validation, regularization

### Risk 3: GPU Access Issues
- **Mitigation**: Have CPU fallback, request GPU access early, use Google Colab as backup

### Risk 4: Poor Model Performance
- **Mitigation**: Start simple, iterate, try multiple architectures, feature engineering

### Risk 5: Time Constraints
- **Mitigation**: Prioritize core functionality, use existing libraries, simplify where possible

---

## 7. Success Criteria

### Minimum Viable Product (MVP)
- [ ] Model trains successfully on A100
- [ ] RMSE and MAE calculated on test set
- [ ] Prediction vs actual plot generated
- [ ] System specifications documented

### Stretch Goals
- [ ] RMSE < 5% of average stock price
- [ ] Directional accuracy > 55%
- [ ] Model works on multiple stocks
- [ ] Ensemble model with improved performance
- [ ] Real-time prediction capability

---

## 8. Next Steps

1. **Immediate**: Fix bugs in `finalex.py` (missing `scaled` variable, date indexing)
2. **Week 1 Start**: Implement data collection and preprocessing pipeline
3. **Week 1 End**: Complete feature engineering and data validation
4. **Week 2 Start**: Design and implement LSTM architecture
5. **Week 2 End**: Set up training on A100 GPU
6. **Week 3**: Train models and tune hyperparameters
7. **Week 4**: Final evaluation, visualization, and documentation

---

## 9. References and Resources

- TensorFlow/Keras LSTM Documentation
- Time Series Forecasting Best Practices
- Stock Market Prediction Papers
- A100 GPU Optimization Guides
- yfinance Documentation
- Technical Analysis Indicators (TA-Lib, pandas_ta)

---

**Last Updated**: December 1, 2025
**Status**: Planning Phase

