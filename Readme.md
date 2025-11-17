# Stock Price Prediction using LSTM / RNN

About
This repository demonstrates a simple end-to-end approach for predicting historical stock closing prices using recurrent neural networks (RNNs), specifically Long Short-Term Memory (LSTM) networks. The included script `pricePredictionLSTM.py` trains an LSTM on historical daily closing prices and produces predictions, visualizations, and a CSV comparing predicted vs actual test prices.

Key details
- Data: Example input file `DIS.csv` (downloadable from Yahoo Finance).
- Example timeframe: 1997-01-01 to 2017-01-01 (training ≈1997–2007, testing ≈2007–2017).
- Model: Single LSTM layer with 25 units → Dropout(0.1) → Dense(1).
- Look-back window: 240 time steps (uses previous 240 days to predict the next day).
- Preprocessing: MinMax scaling to [0, 1].
- Training: Adam optimizer, MSE loss, 1000 epochs, batch size 240 (configurable in the script).
- Outputs:
  - Matplotlib plot comparing actual and predicted series.
  - `lstm_result.csv` containing predicted and actual test prices (rounded).
  - Printed RMSE for train and test sets.

How it works (high level)
1. Load daily closing prices and reshape into a single-feature time series.
2. Normalize values with MinMaxScaler.
3. Convert the sequence into supervised samples: each sample contains `look_back` prior days as input and the next day's closing price as the label.
4. Train an LSTM network on the training split.
5. Predict on training and test sets, invert scaling to original price units, compute RMSE.
6. Save test predictions and actuals to CSV and plot the results.

Intended audience and use cases
- Learners exploring time series forecasting and recurrent neural networks.
- Practitioners building baseline sequence models for stock-price experiments.
- Developers who want a minimal, runnable example to extend (multivariate inputs, hyperparameter tuning, advanced architectures).

Limitations and notes
- Educational demonstration only — not financial advice. Historical performance does not guarantee future results.
- Uses a single feature (closing price). Real-world systems typically use additional features (volume, OHLC, technical indicators, news).
- Training for many epochs by default; consider using validation and callbacks (EarlyStopping, ModelCheckpoint) to avoid overfitting and speed up experiments.
- The script uses a fixed train/test split; prefer walk-forward or time-series cross-validation for robust evaluation.

Suggestions for improvements
- Add CLI arguments or a config file to change dataset, look-back, model hyperparameters, and training options.
- Use a validation set and callbacks to stop training early and save the best model.
- Try stacked LSTMs, GRUs, bidirectional layers, or attention mechanisms.
- Add more input features and feature engineering (indicators, lagged returns, volume).
- Save and load trained model weights; add reproducible seeds and logging.

Credits
- Keras (for neural networks) and scikit-learn (for preprocessing and metrics).
- Example data can be downloaded from Yahoo Finance.

Usage (quick)
- Place a CSV like `DIS.csv` in the repo root (the script expects the closing price column at index 5).
- Run: `python pricePredictionLSTM.py`
- Results: plot display and `lstm_result.csv` with predictions and test prices.

Note: If you want, I can convert this into a full README with a usage section showing common command-line options, or add a short script to parameterize dataset path, look-back, epochs, and model settings.
