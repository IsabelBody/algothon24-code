import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from joblib import Parallel, delayed


# takes around 11 minutes , but for a much better return 

# mean(PL): 46.3
# return: 0.01195
# StdDev(PL): 443.83
# annSharpe(PL): 1.65
# totDvolume: 971388 
# Score: 1.87


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
    lNorm = np.sqrt(np.dot(priceChanges, priceChanges))
    priceChanges /= lNorm
    scaling_factor = 5000
    rpos = np.array([int(x) for x in scaling_factor * priceChanges / latest_price])
    currentPos = np.array([int(x) for x in currentPos + rpos])
    
    return currentPos
