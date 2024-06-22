import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from joblib import Parallel, delayed

import warnings
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

    # Calculate volatility for each instrument
    volatility = np.std(prcSoFar, axis=1)

    # Avoid division by zero and extremely low volatility
    volatility = np.where(volatility == 0, 1, volatility)

    # Normalize price changes by volatility
    risk_adjusted_changes = priceChanges / volatility
    lNorm = np.sqrt(np.dot(risk_adjusted_changes, risk_adjusted_changes))
    risk_adjusted_changes /= lNorm

    # Dynamic scaling factor based on overall market volatility
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

# Assuming `data` is your dataset containing prices for 500 days
data = np.random.rand(nInst, 500)  # replace with actual data

# Splitting the data
train_data = data[:, :400]
val_data = data[:, 400:500]

# Evaluate on training set
train_positions = getMyPosition(train_data)
# Compute metrics on train_positions...

# Evaluate on validation set
val_positions = getMyPosition(val_data)
# Compute metrics on val_positions...

# Compare training and validation performance to detect overfitting

# Finally, evaluate on test set when ready
# test_data = ... (the additional 250 days of data)
# test_positions = getMyPosition(test_data)
# Compute metrics on test_positions...
