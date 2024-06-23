import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')


# mean(PL): 5.2
# return: 0.02673
# StdDev(PL): 23.26
# annSharpe(PL): 3.50
# totDvolume: 48403
# Score: 2.83
# Time: 3m59s90

nInst = 50  # number of instruments (stocks)
currentPos = np.zeros(nInst)

# Cache for ARIMA models to avoid re-fitting
arima_cache = {}

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
    if transformation == 'Original':
        return stock_series
    elif transformation == 'Differencing':
        return np.diff(stock_series)
    elif transformation == 'Log Differencing':
        log_mask = stock_series > 0
        log_series = np.log(stock_series[log_mask])
        return np.diff(log_series)

def fit_arima_model(stock_series, stock_id):
    if stock_id in arima_cache:
        model_fit = arima_cache[stock_id]
    else:
        try:
            model = ARIMA(stock_series, order=(1, 0, 0))  # Simplified ARIMA order
            model_fit = model.fit()
            arima_cache[stock_id] = model_fit
        except:
            return np.nan
    
    forecast = model_fit.forecast(steps=1)[0]
    return forecast

def smooth_series(series, window=5):
    return pd.Series(series).rolling(window, min_periods=1).mean().values

def process_stock(i, prcSoFar):
    stock_prices = prcSoFar[i]
    smoothed_prices = smooth_series(stock_prices)
    best_transformation = determine_best_transformation(smoothed_prices)
    transformed_series = apply_transformation(smoothed_prices, best_transformation)
    predicted_price = fit_arima_model(transformed_series, i)
    return predicted_price

def calculate_scaling_factor(prcSoFar, risk_adjusted_changes):
    recent_performance = np.mean(prcSoFar[:, -10:], axis=1)  # Average performance of the last 10 days
    performance_adjustment = 1 + np.tanh(recent_performance / 1000)  # Dynamic adjustment
    market_volatility = np.std(prcSoFar[:, -1])
    return 5000 * (1 / market_volatility) * performance_adjustment

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

    # Calculate volatility for each instrument
    volatility = np.std(prcSoFar, axis=1)

    # Avoid division by zero and extremely low volatility
    volatility = np.where(volatility == 0, 1, volatility)

    # Normalize price changes by volatility
    risk_adjusted_changes = priceChanges / volatility
    lNorm = np.sqrt(np.dot(risk_adjusted_changes, risk_adjusted_changes))
    risk_adjusted_changes /= lNorm

    # Dynamic scaling factor based on overall market volatility and recent performance
    scaling_factor = calculate_scaling_factor(prcSoFar, risk_adjusted_changes)

    # Calculate desired positions
    rpos = scaling_factor * risk_adjusted_changes / latest_price

    # Apply position limits
    max_positions = 10000 / latest_price
    rpos = np.clip(rpos, -max_positions, max_positions)
    
    new_positions = currentPos + rpos
    currentPos = np.clip(new_positions, -max_positions, max_positions).astype(int)

    # Diversify positions: Reduce positions based on an equal weight strategy
    total_value = np.sum(np.abs(currentPos) * latest_price)
    target_value_per_stock = total_value / nInst
    diversified_positions = np.clip(currentPos, -target_value_per_stock / latest_price, target_value_per_stock / latest_price).astype(int)

    return diversified_positions