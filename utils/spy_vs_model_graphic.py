import matplotlib.pyplot as plt
import numpy as np

# Define the most accurate model's results from the summary (5m-24h prediction)
timeframe = "5m-24h"
real_prices = np.array([580.76, 573.61, 584.84, 581.23, 582.38])  # Actual SPY prices as per the log
predicted_prices = np.array([580.76, 574.39, 584.63, 580.17, 581.98])  # Corresponding predicted prices

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(real_prices, label="Actual Price", marker='o', linestyle='-', color="blue")
plt.plot(predicted_prices, label="Predicted Price (Model)", marker='x', linestyle='--', color="orange")

# Add titles and labels
plt.title(f"SPY Actual vs Predicted Prices - {timeframe} (Most Accurate Model)")
plt.xlabel("Sample")
plt.ylabel("SPY Price")
plt.legend()
plt.grid(True)

# Show plot
plt.show()
