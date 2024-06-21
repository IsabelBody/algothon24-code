import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

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

def fit_arima_model(stock_series, transformation):
    transformed_series = apply_transformation(stock_series, transformation)
    
    # Fit ARIMA model (example, you'll need to tune parameters)
    model = ARIMA(transformed_series, order=(1, 1, 0))  # Example order, tune these parameters
    model_fit = model.fit()
    
    # Forecast next day's price
    forecast = model_fit.forecast(steps=1)[0]
    
    if transformation == 'Differencing':
        forecast += stock_series[-1]
    elif transformation == 'Log Differencing':
        forecast = np.exp(forecast + np.log(stock_series[-1]))
    
    return forecast

def process_stock(i, prcSoFar):
    stock_prices = prcSoFar[i]
    best_transformation = determine_best_transformation(stock_prices)
    predicted_price = fit_arima_model(stock_prices, best_transformation)
    return predicted_price

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape

    if nt < 2:
        return np.zeros(nins)
    
    # Parallel processing for stock predictions
    predictedPrices = Parallel(n_jobs=-1)(delayed(process_stock)(i, prcSoFar) for i in range(nins))
    
    predictedPrices = np.array(predictedPrices)
    latest_price = prcSoFar[:, -1]
    priceChanges = predictedPrices - latest_price
    lNorm = np.sqrt(np.dot(priceChanges, priceChanges))
    priceChanges /= lNorm
    scaling_factor = 5000
    rpos = np.array([int(x) for x in scaling_factor * priceChanges / latest_price])
    currentPos = np.array([int(x) for x in currentPos + rpos])
    
    return currentPos
