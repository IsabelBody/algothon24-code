import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load prices data
def loadPrices(filename):
    df = pd.read_csv(filename, sep='\s+', header=None)
    return df.values.T  # transpose to get nInst x nt

# Predict next day prices using ARIMA
def predict_nextday_prices(prices, order=(5, 1, 0)):
    try:
        model = ARIMA(prices, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return forecast[0]
    except:
        # In case ARIMA fails, return the last available price as a naive forecast
        return prices[-1]

# Plot actual vs. predicted prices for selected stocks
def plot_predictions(prcSoFar, stock_indices, start_day=200):
    for i in stock_indices:
        actual_prices = prcSoFar[i, start_day:]
        predicted_prices = []
        
        for t in range(start_day, prcSoFar.shape[1]):
            prc_hist = prcSoFar[i, :t]
            pred_price = predict_nextday_prices(prc_hist)
            predicted_prices.append(pred_price)
        
        days = range(start_day, prcSoFar.shape[1])
        plt.figure(figsize=(12, 6))
        plt.plot(days, actual_prices, label='Actual Prices')
        plt.plot(days, predicted_prices, label='Predicted Prices')
        plt.title(f'Stock {i + 1} Prices')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

# Main function to execute the process
def main():
    prices_file = "./prices.txt"  # Update the path if necessary
    prcAll = loadPrices(prices_file)
    print(f"Loaded {prcAll.shape[0]} instruments for {prcAll.shape[1]} days")
    
    selected_stocks = range(5)  # Select first 10 stocks for testing
    plot_predictions(prcAll, selected_stocks, start_day=200)

if __name__ == "__main__":
    main()
