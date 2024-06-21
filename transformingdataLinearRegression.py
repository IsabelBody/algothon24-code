import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller

nInst = 50  # number of instruments (stocks)
currentPos = np.zeros(nInst)

def adf_test(series):
    result = adfuller(series)
    return result[1]  # returning the p-value

def determine_best_transformation(stock_series):
    # Original series
    pval_original = adf_test(stock_series)
    
    # First difference
    diff_series = stock_series.diff().dropna()
    pval_diff = adf_test(diff_series)
    
    # Log transformation and differencing
    log_series = np.log(stock_series[stock_series > 0])  # Log transformation requires positive values
    log_diff_series = log_series.diff().dropna()
    pval_log_diff = adf_test(log_diff_series)
    
    # Determine the best transformation
    pvals = [pval_original, pval_diff, pval_log_diff]
    transformations = ['Original', 'Differencing', 'Log Differencing']
    
    best_transformation = transformations[np.argmin(pvals)]
    return best_transformation

def apply_transformation(stock_series, transformation):
    if transformation == 'Original':
        return stock_series
    elif transformation == 'Differencing':
        return stock_series.diff().dropna()
    elif transformation == 'Log Differencing':
        return np.log(stock_series[stock_series > 0]).diff().dropna()

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

def determine_best_transformation(stock_series):
    pval_original = adf_test(stock_series)
    
    diff_series = stock_series.diff().dropna()
    pval_diff = adf_test(diff_series)
    
    if pval_original < 0.05:
        return 'Original'
    
    log_mask = stock_series > 0
    log_series = np.log(stock_series[log_mask])
    log_diff_series = log_series.diff().dropna()
    pval_log_diff = adf_test(log_diff_series)
    
    pvals = [pval_original, pval_diff, pval_log_diff]
    transformations = ['Original', 'Differencing', 'Log Differencing']
    
    best_transformation = transformations[np.argmin(pvals)]
    return best_transformation

def apply_transformation(stock_series, transformation):
    if transformation == 'Original':
        return stock_series
    elif transformation == 'Differencing':
        diff_series = stock_series.diff().dropna()
        return diff_series
    elif transformation == 'Log Differencing':
        log_mask = stock_series > 0
        log_series = np.log(stock_series[log_mask])
        log_diff_series = log_series.diff().dropna()
        return log_diff_series

def predict_nextday_prices(prices, transformation):
    transformed_prices = apply_transformation(prices, transformation)
    if transformed_prices is None or transformed_prices.empty:
        return prices.iloc[-1]  # Fall back to the last known price if transformation fails

    features = get_features(transformed_prices)
    X = np.arange(len(features)).reshape(-1, 1)
    y = features['short_mavg'].values  # Use .values for numpy array
    
    model = LinearRegression()
    model.fit(X, y)
    
    next_day = np.array([[len(features)]])
    next_day_prediction = model.predict(next_day)
    
    if transformation == 'Differencing':
        next_day_prediction += prices.iloc[-1]
    elif transformation == 'Log Differencing':
        next_day_prediction = np.exp(next_day_prediction + np.log(prices.iloc[-1]))
    
    return next_day_prediction[0]

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape

    if nt < 2:
        return np.zeros(nins)
    
    predictedPrices = np.zeros(nins)
    transformations = []
    
    for i in range(nins):
        stock_prices = pd.Series(prcSoFar[i])
        best_transformation = determine_best_transformation(stock_prices)
        transformations.append(best_transformation)
        predictedPrices[i] = predict_nextday_prices(stock_prices, best_transformation)
    
    latest_price = prcSoFar[:, -1]
    priceChanges = predictedPrices - latest_price
    lNorm = np.sqrt(np.dot(priceChanges, priceChanges))
    priceChanges /= lNorm
    scaling_factor = 5000
    rpos = np.array([int(x) for x in scaling_factor * priceChanges / latest_price])
    currentPos = np.array([int(x) for x in currentPos + rpos])
    
    return currentPos
