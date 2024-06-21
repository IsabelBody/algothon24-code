import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA

nInst = 50  # number of instruments (stocks)
currentPos = np.zeros(nInst)

def get_features(prices):
    # Compute moving averages
    short_window = 20
    long_window = 100
    signals = pd.DataFrame(index=prices.index)
    signals['short_mavg'] = prices.rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = prices.rolling(window=long_window, min_periods=1).mean()
    
    # Compute momentum
    signals['momentum'] = prices.diff(3)
    
    # Compute volatility
    signals['volatility'] = prices.rolling(window=short_window, min_periods=1).std()
    
    return signals.dropna()

def predict_nextday_prices(prices):
    features = get_features(prices)
    X = np.arange(len(features)).reshape(-1, 1)
    y = features['short_mavg']
    
    model = LinearRegression()
    model.fit(X, y)
    
    next_day = np.array([[len(features)]])
    next_day_prediction = model.predict(next_day)
    
    return next_day_prediction[0]

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape

    if (nt < 2):
        return np.zeros(nins)
    
    predictedPrices = np.zeros(nins)
    for i in range(nins):
        stock_prices = pd.Series(prcSoFar[i])
        predictedPrices[i] = predict_nextday_prices(stock_prices)
    
    latest_price = prcSoFar[:, -1]
    priceChanges = predictedPrices - latest_price
    lNorm = np.sqrt(priceChanges.dot(priceChanges))
    priceChanges /= lNorm
    scaling_factor = 5000
    rpos = np.array([int(x) for x in scaling_factor * priceChanges / latest_price])
    currentPos = np.array([int(x) for x in currentPos + rpos])
    return currentPos

