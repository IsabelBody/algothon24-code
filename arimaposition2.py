import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')



# mean(PL): 10.6
# return: 0.01969
# StdDev(PL): 50.30
# annSharpe(PL): 3.33
# totDvolume: 134857
# Score: 5.55
# time: 7m38s07





nInst = 50  # number of instruments (stocks)
currentPos = np.zeros(nInst)

def adf_test(series):
    return adfuller(series, autolag='AIC')[1]  # returning the p-value

def determine_best_transformation(stock_series):
    pval_original = adf_test(stock_series)
    if pval_original < 0.05:
        return 'Original'
    
    diff_series = np.diff(stock_series)
    pval_diff = adf_test(diff_series)
    if pval_diff < 0.05:
        return 'Differencing'
    
    log_mask = stock_series > 0
    if not np.any(log_mask):
        return 'Differencing'
    
    log_series = np.log(stock_series[log_mask])
    log_diff_series = np.diff(log_series)
    pval_log_diff = adf_test(log_diff_series)
    if pval_log_diff < 0.05:
        return 'Log Differencing'
    
    return 'Differencing'  # Default if none are stationary

def apply_transformation(stock_series, transformation):
    if transformation == 'Original':
        return stock_series
    elif transformation == 'Differencing':
        return np.diff(stock_series)
    elif transformation == 'Log Differencing':
        log_series = np.log(stock_series[stock_series > 0])
        return np.diff(log_series)

def fit_arima_model(stock_series):
    try:
        model = ARIMA(stock_series, order=(1, 0, 0))  # Simplified ARIMA order
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)[0]
        return forecast
    except Exception:
        return np.nan

def process_stock(stock_prices):
    best_transformation = determine_best_transformation(stock_prices)
    transformed_series = apply_transformation(stock_prices, best_transformation)
    return fit_arima_model(transformed_series)

def getMyPosition(prcSoFar):
    global currentPos
    nInst, nt = prcSoFar.shape

    if nt < 2:
        return np.zeros(nInst)
    
    # Parallel processing for stock predictions
    predictedPrices = Parallel(n_jobs=-1)(delayed(process_stock)(prcSoFar[i]) for i in range(nInst))
    
    predictedPrices = np.array(predictedPrices)
    latest_price = prcSoFar[:, -1]
    priceChanges = predictedPrices - latest_price

    # Calculate volatility for each instrument
    volatility = np.std(prcSoFar, axis=1, ddof=1)

    # Avoid division by zero and extremely low volatility
    volatility = np.where(volatility == 0, 1, volatility)

    # Normalize price changes by volatility
    risk_adjusted_changes = priceChanges / volatility
    lNorm = np.linalg.norm(risk_adjusted_changes)
    risk_adjusted_changes /= lNorm

    # Dynamic scaling factor based on overall market volatility
    market_volatility = np.std(latest_price, ddof=1)
    scaling_factor = 5000 * (1 / market_volatility)

    # Calculate desired positions
    rpos = scaling_factor * risk_adjusted_changes / latest_price

    # Apply position limits
    max_positions = 10000 / latest_price
    rpos = np.clip(rpos, -max_positions, max_positions)
    
    new_positions = currentPos + rpos
    currentPos = np.clip(new_positions, -max_positions, max_positions).astype(int)

    return currentPos
