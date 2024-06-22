import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
from joblib import Parallel, delayed

# Takes 30 minutes 
# mean(PL): 56.4
# return: 0.00881
# StdDev(PL): 387.85
# annSharpe(PL): 2.30
# totDvolume: 1606160
# Score: 17.62



warnings.filterwarnings('ignore')

nInst = 50  # number of instruments (stocks)
currentPos = np.zeros(nInst)

def adf_test(series):
    result = adfuller(series)
    return result[1]  # returning the p-value

def determine_best_transformation(stock_series, stock_index):
    pval_original = adf_test(stock_series)
    is_stationary_original = pval_original < 0.05
    
    diff_series = np.diff(stock_series)
    pval_diff = adf_test(diff_series)
    is_stationary_diff = pval_diff < 0.05
    
    log_mask = stock_series > 0
    log_series = np.log(stock_series[log_mask])
    log_diff_series = np.diff(log_series)
    pval_log_diff = adf_test(log_diff_series)
    is_stationary_log_diff = pval_log_diff < 0.05
        
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

def fit_arima_model(stock_series, transformation):
    transformed_series = apply_transformation(stock_series, transformation)
    
    # Simplified ARIMA parameter selection
    orders = [(1, 0, 0), (1, 1, 0), (0, 1, 1)]
    best_aic = np.inf
    best_order = None
    best_model = None
    
    for order in orders:
        try:
            model = ARIMA(transformed_series, order=order)
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order = order
                best_model = model_fit
        except:
            continue
    
    # Forecast next day's price using the best model
    forecast = best_model.forecast(steps=1)[0]
    
    if transformation == 'Differencing':
        forecast += stock_series[-1]
    elif transformation == 'Log Differencing':
        forecast = np.exp(forecast + np.log(stock_series[-1]))
    
    return forecast

def process_stock(i, prcSoFar):
    stock_prices = prcSoFar[i]
    best_transformation = determine_best_transformation(stock_prices, i)
    predicted_price = fit_arima_model(stock_prices, best_transformation)
    return predicted_price

def getMyPosition(prcSoFar):
    global currentPos
    (nInst, nt) = prcSoFar.shape

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
