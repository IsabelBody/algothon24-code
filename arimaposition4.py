import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from joblib import Parallel, delayed
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# mean(PL): 25.4
# return: 0.03480
# StdDev(PL): 129.70
# annSharpe(PL): 3.09
# totDvolume: 183000
# Score: 12.40


nInst = 50  # number of instruments (stocks)
currentPos = np.zeros(nInst)

def adf_test(series):
    result = adfuller(series)
    return result[1]  # returning the p-value

def determine_best_transformation(stock_series):
    pval_original = adf_test(stock_series)
    
    diff_series = np.diff(stock_series)
    pval_diff = adf_test(diff_series)
    
    if pval_original < 0.05:
        return 'Original'
    
    log_mask = stock_series > 0
    log_series = np.log(stock_series[log_mask])
    log_diff_series = np.diff(log_series)
    pval_log_diff = adf_test(log_diff_series)
    
    pvals = [pval_original, pval_diff, pval_log_diff]
    transformations = ['Original', 'Differencing', 'Log Differencing']
    
    best_transformation = transformations[np.argmin(pvals)]
    return best_transformation

def apply_transformation(stock_series, transformation):
    if (len(stock_series) < 2):
        return stock_series
    if transformation == 'Original':
        return stock_series
    elif transformation == 'Differencing':
        return np.diff(stock_series)
    elif transformation == 'Log Differencing':
        log_mask = stock_series > 0
        log_series = np.log(stock_series[log_mask])
        return np.diff(log_series)

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
    best_transformation = determine_best_transformation(stock_prices)
    transformed_series = apply_transformation(stock_prices, best_transformation)
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

    # Calculate EMA for each instrument as a volatility measure
    def ema(series, span=20):
        return series.ewm(span=span, adjust=False).mean().values[-1]
    
    volatilities = np.array([ema(pd.Series(prcSoFar[i])) for i in range(nInst)])

    # Avoid division by zero and extremely low volatility
    volatilities = np.where(volatilities == 0, 1, volatilities)

    # Normalize price changes by volatility
    risk_adjusted_changes = priceChanges / volatilities
    lNorm = np.sqrt(np.dot(risk_adjusted_changes, risk_adjusted_changes))
    risk_adjusted_changes /= lNorm

    # Calculate dynamic scaling factor based on overall market volatility
    market_volatility = np.std(latest_price)
    scaling_factor = 5000 * (1 / market_volatility)

    # Calculate desired positions
    rpos = scaling_factor * risk_adjusted_changes / latest_price

    # Apply position limits
    max_positions = 10000 / latest_price
    rpos = np.clip(rpos, -max_positions, max_positions)
    
    new_positions = currentPos + rpos
    currentPos = np.clip(new_positions, -max_positions, max_positions).astype(int)

    return currentPos
