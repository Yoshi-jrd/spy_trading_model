# SPY Trading Model Project: Rough Development Roadmap

### **Phase 1: Data Collection and Preparation (Completed)**

- **Data Loader Development**:

  - Pull real-time and historical data (SPY price, options data, sentiment, IV, VIX, etc.).
  - Ensure proper formatting and validation of pulled data.
  - Cache historical data for backtesting purposes.
  - Add data processing logic (cleaning NaN values, ensuring datetime compatibility).
  - Debugging data corruptions and handling inconsistencies.

- **Current Status**:
  - Data pull is working correctly, and the data corruption issue has been resolved.
  - Implemented the logic to append new data rather than overwriting historical data.

### **Phase 2: Model Framework Setup**

- **Model Development**:
  - **RandomForest, XGBoost, GradientBoosting**: Establish models for shorter timeframes.
  - **LSTM**: For long-term predictions and capturing sequential data patterns.
- **Key Tasks**:

  - **Test individual models**:

    - Validate that each model (RandomForest, XGBoost, GradientBoosting, LSTM) imports the data correctly and processes it.
    - Verify that they are properly learning and updating with new data inputs.
    - Ensure they are producing valid outputs and predictions based on the provided data.

  - **Data Processing Checkpoints**:
    - Ensure that each model handles all the technical indicators (MACD, RSI, etc.) and options data (IV, volume, open interest) correctly.
    - Check for missing or corrupt values in the input during the testing process.
    - Output key debugging information (dataset heads, tails, datetime validation).

### **Phase 3: Model Hypertuning**

- **Hypertuning the Models**:

  - Once the individual models are verified, proceed with hypertuning.
  - Optimize hyperparameters for each model to improve prediction accuracy (e.g., tuning depth, learning rates, number of estimators).
  - Use cross-validation and the hypertuning scripts to find the best parameter sets for each model.

- **Key Tasks**:
  - Ensure that hypertuning works across all models.
  - Output results to review performance for each set of hyperparameters.
  - Select the optimal models for integration into the main predictive workflow.

### **Phase 4: Model Evaluation and Trade Simulations**

- **Model Evaluation**:

  - Assess the individual model performances using historical data.
  - Verify that model stacking (RandomForest, XGBoost, GradientBoosting) is working as intended to improve accuracy.
  - Validate that LSTM is performing well for 1-hour and 1-day timeframes.

- **Simulated Trade Execution**:
  - Run trade simulations based on model outputs.
  - Implement simulation logic for different trade types: credit spreads, debit spreads, condors, straddles.
  - Incorporate risk management strategies (stop loss, profit-taking triggers).

### **Phase 5: Live Trading and Automation**

- **Automation of Trade Execution**:

  - Finalize trade execution logic based on model predictions.
  - Integrate rolling and early closure logic for trade management.
  - Test and verify the model's performance in live market conditions.

- **Ongoing Enhancements**:
  - Continue refining model predictions based on new data and trade results.
  - Add additional strategies (butterflies, strangles, etc.) if volatility or other conditions make them favorable.

---

### **Current Focus (Phase 2):**

1. **Individual Model Testing**:

   - Ensure each model pulls and processes the data correctly.
   - Validate that RandomForest, XGBoost, GradientBoosting, and LSTM are handling data inputs and producing valid predictions.

2. **Hypertuning**:
   - Once testing is complete, begin hypertuning to optimize model performance.
