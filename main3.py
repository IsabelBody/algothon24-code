import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

# Takes just below 10 minutes 

# mean(PL): 38.3
# return: 0.00974
# StdDev(PL): 447.44
# annSharpe(PL): 1.35
# totDvolume: 986408
# Score: -6.48

nInst = 50  # number of instruments (stocks)
currentPos = np.zeros(nInst)

def apply_differencing(stock_series):
    return np.diff(stock_series)

def fit_arima_model(stock_series):
    try:
        model = ARIMA(stock_series, order=(1, 0, 0))  # Simplified ARIMA order
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)[0]
        return forecast
    except:
        return np.nan

def process_stock(i, prcSoFar):
    stock_prices = prcSoFar[i]
    transformed_series = apply_differencing(stock_prices)
    predicted_price = fit_arima_model(transformed_series)
    return predicted_price

def getMyPosition(prcSoFar):
    global currentPos
    nInst, nt = prcSoFar.shape

    if nt < 2:
        return np.zeros(nInst)
    
    # Parallel processing for stock predictions
    predictedPrices = Parallel(n_jobs=-1)(delayed(process_stock)(i, prcSoFar) for i in range(nInst))
    
    predictedPrices = np.array(predictedPrices)
    latest_price = prcSoFar[:, -1]
    priceChanges = predictedPrices - latest_price
    lNorm = np.sqrt(np.dot(priceChanges, priceChanges))
    priceChanges /= lNorm
    scaling_factor = 5000
    rpos = np.array([int(x) for x in scaling_factor * priceChanges / latest_price])
    currentPos = np.array([int(x) for x in currentPos + rpos])
    
    return currentPos

# Example usage
# prices = np.random.randn(50, 200)  # Example price data
# print(getMyPosition(prices))
