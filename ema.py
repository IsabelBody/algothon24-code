import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import multiprocessing as mp

def get_ema(prices, span):
    return prices.ewm(span=span, adjust=False).mean()

def predict_nextday_prices_ema(prices):
    span = 10
    ema = get_ema(prices, span)
    return ema.iloc[-1]

def predict_nextday_prices_linear(prices):
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices.values
    model = LinearRegression()
    model.fit(X, y)
    next_day = np.array([[len(prices)]])
    next_day_prediction = model.predict(next_day)
    return next_day_prediction[0]

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape

    if nt < 2:
        return np.zeros(nins)
    
    predictedPrices = np.zeros(nins)
    for i in range(nins):
        stock_prices = pd.Series(prcSoFar[i])
        ema_prediction = predict_nextday_prices_ema(stock_prices)
        linear_prediction = predict_nextday_prices_linear(stock_prices)
        predictedPrices[i] = (ema_prediction + linear_prediction) / 2
    
    latest_price = prcSoFar[:, -1]
    priceChanges = predictedPrices - latest_price
    lNorm = np.sqrt(priceChanges.dot(priceChanges))
    priceChanges /= lNorm
    scaling_factor = 5000
    rpos = (scaling_factor * priceChanges / latest_price).astype(int)
    
    currentPos = (currentPos + rpos).astype(int)
    return currentPos

# Initialize currentPos
currentPos = np.zeros(50)
