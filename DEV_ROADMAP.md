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

# Current State Improvements and immediate next steps

### Review of Current Code:

- **Data Loader (`data_loader.py`)**:

  - Properly imports SPY data and additional datasets (VIX, IV, GDP, CPI, Sentiment, Articles).
  - Cleans the data using `ensure_datetime()` and handles missing values with forward and backward filling.
  - Saves the data into a pickle file (`historical_data.pickle`).
  - Ensures that new data overwrites the existing data properly (not just appending).

- **Training Models (`train_models.py`)**:
  - Trains models using multiple timeframes (`5m`, `15m`, `1h`, `1d`) of SPY data.
  - Uses the best parameters for `RandomForest`, `XGBoost`, `GradientBoosting`, and `LSTM`.
  - The models are hypertuned dynamically, with RMSE being tracked to update and store the best parameters.
  - Evaluates model performance using MAE and RMSE.
  - LSTM is now correctly handling reshaping of input data for time-series analysis.

### Changes and Improvements Made:

1. **MACD Histogram Calculations**:

   - The `MACD`, `MACD_Signal`, and `MACD_Histogram` values are now correctly calculated and loaded into the dataset.
   - The issue with `NaN` values for these indicators has been resolved.

2. **Impulse MACD**:

   - The color-coding (`Impulse_Color`) is now functional and reflects market sentiment changes based on MACD and RSI.

3. **Overwriting vs. Appending**:

   - We ensured that data is properly overwritten (instead of incorrectly appending new data), which resolves issues with data duplication.
   - We’ve maintained a clean and updated `pickle` file (`historical_data.pickle`).

4. **Data Integrity**:

   - All datasets (SPY, VIX, IV, GDP, CPI, Sentiment, Articles) are now loaded, cleaned, and normalized properly.
   - No critical issues with missing or misaligned data.

5. **Model Training and Hypertuning**:

   - The code dynamically updates the best parameters based on RMSE, storing them to `best_params.pkl`.
   - We have ensured that all models (RandomForest, XGBoost, GradientBoosting, LSTM) are trained across all SPY timeframes with the calculated indicators.

6. **Indicator Calculation**:

   - Indicators like MACD, RSI, Bollinger Bands, ATR, and others are integrated into the model training process.

7. **LSTM Model Fixes**:

   - LSTM input reshaping and handling are fixed for time-series data.
   - LSTM now trains successfully without issues related to input dimensions.

8. **Performance Enhancements**:
   - Refactoring for cleaner, more modular code across data loading, model training, and evaluation.
   - Performance optimization with RMSE-based dynamic parameter updates.

### Are We Ready for Additional Training?

Yes, the model is ready for additional training and testing with all improvements. The system now:

- Properly handles data input, normalization, and indicator calculation.
- Uses dynamic hypertuning based on RMSE to find the best model parameters.
- Can efficiently train and evaluate models across multiple SPY timeframes.

### List of Improvements:

1. Resolved `NaN` issues in `MACD` and `Impulse_MACD`.
2. Added proper color coding for `Impulse_MACD` and fixed calculation issues.
3. Switched from appending data to overwriting, ensuring data consistency.
4. Improved data loading, cleaning, and normalizing steps for all datasets.
5. Integrated indicators (MACD, RSI, Bollinger Bands, etc.) with the model training process.
6. Added proper hypertuning for all models using RMSE to track and update best parameters.
7. Fixed LSTM input handling for time-series data.
8. Modularized code for better organization and performance optimization.
9. Verified that data is properly saved to and loaded from the pickle file, ensuring clean, historical data is used.

### Next Steps:

- Train the models with the latest cleaned data.
- Track and monitor performance, adjusting for any improvements as needed.
- Review results and ensure that predictions are accurate and consistent across different timeframes.
