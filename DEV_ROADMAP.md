# SPY Trading Model Project: Comprehensive Development Roadmap

---

## Project Overview

- **Objective**: Build a predictive SPY options trading model leveraging various machine learning techniques to maximize profitability in short-term options trades.
- **Core Strategies**: Focus on credit spreads, debit spreads, condors, and straddles. Utilize models to predict price action and guide trade entries and exits.
- **Programming Language**: Python
- **Primary Frameworks**: TensorFlow/Keras, Scikit-learn, XGBoost, and PyTorch (potential future use for enhanced deep learning models)
- **File Structure**: Modularized for scalability and easy debugging.

---

## Phase 1: Data Collection and Preparation (Completed)

### **Modules and Key Files**

1. **`data_loader.py`**: Handles data extraction, cleaning, and caching.

   - Sources: SPY price, options chain, IV, VIX, sentiment, GDP, CPI.
   - Normalizes data formats, cleans NaNs, and ensures datetime compatibility.
   - Caches historical data to `historical_data.pickle` for consistent backtesting.

2. **`data_storage.py`**: Manages data saving and loading operations for efficient access.

   - Stores cleaned, appended data to ensure up-to-date historical datasets.

3. **`indicator_calculator.py`**: Calculates technical indicators for SPY data.

   - **Indicators Included**: MACD, RSI, ATR, ADX, EMA, Bollinger Bands, OBV, MFI.
   - Handles color-coding for `Impulse_MACD` and other sentiment indicators.

4. **Status**: Data pulls are fully functional. All modules work seamlessly, and historical data is accessible via `historical_data.pickle`.

---

## Phase 2: Model Development and Initial Testing (Completed)

### **Modules and Key Files**

1. **`train_models.py`**: Trains individual models (RandomForest, XGBoost, GradientBoosting, LSTM) across multiple timeframes.
   - **Timeframes**: `5m`, `15m`, `1h`, `1d`.
   - **Indicators Used**: MACD, RSI, Bollinger Bands, ATR, ADX, OBV, MFI, Impulse MACD.
   - **Best Parameter Tracking**: RMSE-based tracking, saved in `best_params.pkl`.
2. **Models and their Functions**

   - **RandomForest, XGBoost, GradientBoosting**:
     - Short-term prediction models focusing on 5m, 15m, and 1h timeframes.
   - **LSTM**:
     - For sequential data, specifically targeted at the 1h and 1d timeframes to capture longer-term dependencies.

3. **Testing Outputs**:
   - **Metrics**: Mean Absolute Error (MAE), Root Mean Square Error (RMSE).
   - **Debugging**: Key checkpoint outputs for model performance, data integrity, and input structure validation.

---

## Phase 3: Hypertuning and Model Optimization (In Progress)

### **Goals**: Refine model accuracy through hyperparameter tuning.

1. **Modules and Key Files**:

   - **`hypertuning.py`**:
     - Scripts for cross-validation and parameter optimization across RandomForest, XGBoost, GradientBoosting, and LSTM.
     - Dynamically updates best-performing parameters to `best_params.pkl`.

2. **Tasks**:

   - **Run Hypertuning**:
     - Test multiple parameter combinations to optimize each model’s performance (depth, estimators, learning rates, LSTM layers).
     - Review and record parameter results to finalize models for further evaluation.
   - **Track Results**:
     - Log RMSE and MAE results to determine best-performing configurations for each timeframe.

3. **Next Steps**:
   - Complete hypertuning on the LSTM model for 1h and 1d timeframes.
   - Finalize GradientBoosting and XGBoost configurations based on recent testing results.

---

## Phase 4: Model Evaluation and Backtesting

### **Objective**: Verify model accuracy on historical data, validate stacking methods, and simulate trades.

1. **Modules and Key Files**:

   - **`run_evaluation_workflow.py`**: Coordinates end-to-end evaluation across all models.
   - **`simulate_trade.py`**: Simulates different options strategies based on model predictions.
   - **`evaluate_model.py`**: Calculates and logs MAE and RMSE metrics across different model configurations and timeframes.

2. **Key Tasks**:

   - **Model Evaluation**:
     - Assess MAE and RMSE for each model, confirming consistency in prediction accuracy across timeframes.
   - **Simulated Trade Execution**:
     - Simulate trades, leveraging predicted price ranges and integrating stop-loss and profit-taking triggers.
     - Strategies: Credit spreads, debit spreads, condors, and straddles.
   - **Model Stacking**:
     - Test stacked model predictions (RandomForest, XGBoost, GradientBoosting) to enhance prediction reliability.

3. **Next Steps**:
   - Execute comprehensive backtesting to verify model accuracy and adjust parameters if necessary.
   - Integrate stacked predictions into simulations and analyze the effect on simulated trades.

---

## Phase 5: Real-Time Trade Execution and Automation

### **Objective**: Implement and test live market trade execution based on model outputs.

1. **Modules and Key Files**:

   - **`live_trade_executor.py`**: Responsible for triggering live trades based on model signals.
     - Supports entry, exit, and rolling logic for active trades.
     - Includes stop-loss and profit-taking mechanisms.

2. **Trade Logic**:

   - **Trade Triggering**:
     - Automatically execute high-confidence trades based on model outputs, starting with credit spreads and straddles.
   - **Risk Management**:
     - Set conservative stop-loss levels for live trading.
     - Automate trade rolling and closure based on market conditions to limit exposure.

3. **Real-Time Testing**:
   - Begin low-risk trades in real-time, tracking model predictions against actual price movements.
   - Fine-tune live trade parameters based on initial results, adjusting for market volatility and news events.

---

## Current Status and Immediate Focus

1. **Completed Steps**:

   - **Data Loading**: Ensured all data sources are clean and integrated.
   - **Model Training**: Verified initial training and predictions for all models across multiple timeframes.
   - **LSTM Model Fixes**: Addressed reshaping and data structure requirements.

2. **Current Immediate Focus**:

   - **Finalize Hypertuning**:
     - Complete parameter optimization for all models, prioritizing RMSE improvement.
   - **Backtesting and Evaluation**:
     - Run evaluation workflow on optimized models to verify predictions.
   - **Prepare for Real-Time Testing**:
     - Finalize `live_trade_executor.py` and ensure all trade strategies are correctly triggered.

3. **Performance Enhancements**:
   - Continue code refactoring to maintain modularity and efficiency.
   - Set up enhanced error logging for trade simulations and live executions.

---

## Directory Structure Overview

```plaintext
├── data/
│   ├── data_loader.py          # Loads and processes data
│   ├── data_storage.py         # Manages data saving/loading
│   ├── historical_data.pickle  # Cached historical data
│   └── indicator_calculator.py # Calculates indicators (MACD, RSI, etc.)
├── models/
│   ├── train_models.py         # Main training script for all models
│   ├── random_forest_model.py  # RandomForest model implementation
│   ├── xgboost_model.py        # XGBoost model implementation
│   ├── gradient_boosting_model.py # Gradient Boosting implementation
│   ├── lstm_model.py           # LSTM model setup and training
│   └── hypertuning.py          # Hypertuning scripts for all models
├── simulate/
│   ├── simulate_trade.py       # Simulates trades based on model predictions
│   └── evaluate_model.py       # Evaluates model predictions for accuracy
├── live/
│   ├── live_trade_executor.py  # Executes live trades based on model outputs
├── utils/
│   ├── ensure_datetime.py      # Ensures correct datetime format in data
│   └── model_utils.py          # Utility functions for model evaluation
└── best_params.pkl             # Stores the best parameters based on RMSE

## Future Enhancements

- **Advanced Model Integration**:
    - Explore advanced architectures (e.g., attention-based models) for improved prediction of SPY options.
    - Potentially integrate reinforcement learning for more dynamic trade adjustments.

- **Expanded Indicator Use**:
    - Add additional sentiment indicators or external economic indicators to enhance trade context.

- **Comprehensive Dashboard**:
    - Create a real-time dashboard to track model performance, indicators, and live trade stats.

- **Automated Reporting**:
    - Generate daily/weekly reports on model accuracy, trade performance, and potential adjustments.

---

## Key Next Steps Summary

- **Complete Hypertuning** for all models and finalize optimal parameters.
- **Evaluate Models in Backtesting** to verify predictions and run simulations.
- **Prepare `live_trade_executor.py`** for real-time trading, ensuring seamless integration with trade strategies and risk management.
```
