import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# Differencing function
def difference(data, interval=1):
    return np.diff(data, n=interval)

# Preparing data with log transformation and differencing
def prepare_data(stock_prices):
    print("Applying log transformation...")
    prices_log = np.log(stock_prices)
    
    print("Applying differencing to make the series stationary...")
    prices_diff = difference(prices_log, 1)
    
    return prices_diff, prices_log

# ARIMA prediction function
def predict_nextday_prices_arima(prices):
    print("Preparing data for ARIMA...")
    prepared_prices, original_prices_log = prepare_data(prices)
    
    print("Fitting ARIMA model...")
    model = ARIMA(prepared_prices, order=(5, 1, 0))  # Example order, tune as needed
    model_fit = model.fit()
    
    print("Forecasting next day price...")
    forecast = model_fit.forecast(steps=1)
    
    # Reverse transformation
    next_day_prediction = np.exp(forecast[0] + original_prices_log[-1])
    return next_day_prediction

# Main function to get positions
def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape

    if nt < 2:
        print("Not enough data to make predictions. Returning initial positions.")
        return np.zeros(nins)
    
    predictedPrices = np.zeros(nins)
    print("Predicting prices for each instrument...")
    for i in range(nins):
        print(f"Processing instrument {i+1}/{nins}...")
        stock_prices = prcSoFar[i]
        predictedPrices[i] = predict_nextday_prices_arima(stock_prices)
    
    latest_price = prcSoFar[:, -1]
    priceChanges = predictedPrices - latest_price
    lNorm = np.sqrt(priceChanges.dot(priceChanges))
    priceChanges /= lNorm
    scaling_factor = 5000
    rpos = (scaling_factor * priceChanges / latest_price).astype(int)
    
    currentPos = (currentPos + rpos).astype(int)
    print("Returning updated positions.")
    return currentPos

# Initialize currentPos
currentPos = np.zeros(50)
