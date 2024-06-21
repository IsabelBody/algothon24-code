import pandas as pd

import warnings
from statsmodels.tsa.arima.model import ARIMA


# Suppress specific warnings
warnings.filterwarnings("ignore")


data_path = 'prices.txt'
prices_df = pd.read_csv(data_path, delim_whitespace=True, header=None)

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller



# Function to perform ADF test
def adf_test(series):
    result = adfuller(series)
    return result[1]  # returning the p-value

# Initialize results dictionary
transformation_results = []

# Iterate over each stock
for i in range(prices_df.shape[1]):
    stock_series = prices_df.iloc[:, i]
    
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
    best_pval = min(pvals)
    
    transformation_results.append({
        'Stock': i + 1,
        'Best Transformation': best_transformation,
        'p-value': best_pval,
        'Stationary': 'Yes' if best_pval < 0.05 else 'No'
    })

# Create a DataFrame with the results
transformation_results_df = pd.DataFrame(transformation_results)

# Print the results
print(transformation_results_df)





# Function to fit ARIMA model and predict the next day's price
def fit_arima_and_predict(transformed_series, order=(1, 1, 1)):
    model = ARIMA(transformed_series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)
    return forecast[0]

def get_transformed_series(series, transformation_type):
    if transformation_type == 'Original':
        return series
    elif transformation_type == 'Differencing':
        return series.diff().dropna()
    elif transformation_type == 'Log Differencing':
        log_series = np.log(series[series > 0])
        return log_series.diff().dropna()
    else:
        raise ValueError("Unknown transformation type")

def getMyPosition(prcSoFar, num_instruments=None, num_days=None):
    global currentPos
    
    (nins, nt) = prcSoFar.shape
    
    # Adjust for testing with fewer instruments and days
    if num_instruments is not None and num_instruments < nins:
        prcSoFar = prcSoFar[:num_instruments, :]
        nins = num_instruments
    
    if num_days is not None and num_days < nt:
        prcSoFar = prcSoFar[:, :num_days]
        nt = num_days
    
    if nt < 2:
        return np.zeros(nins)  # Insufficient data to predict
    
    # Prepare to store predicted prices
    predictedPrices = np.zeros(nins)
    
    # Fit ARIMA and predict next day's price for each instrument
    for i in range(nins):
        stock_series = prcSoFar[i, :]
        
        # Get best transformation type from your results
        best_transformation = transformation_results_df.iloc[i]['Best Transformation']
        
        # Transform the series
        transformed_series = get_transformed_series(pd.Series(stock_series), best_transformation)
        
        # Fit ARIMA model and predict
        try:
            prediction = fit_arima_and_predict(transformed_series)
        except:
            prediction = stock_series[-1]  # If ARIMA fails, fallback to last price
        
        # Reverse the transformation if needed (e.g., if log differenced)
        if best_transformation == 'Log Differencing':
            last_price = stock_series[-1]
            predictedPrices[i] = np.exp(np.log(last_price) + prediction)
        else:
            predictedPrices[i] = prediction
    
    latest_price = prcSoFar[:, -1]
    priceChanges = predictedPrices - latest_price
    
    lNorm = np.sqrt(priceChanges.dot(priceChanges))
    priceChanges /= lNorm
    
    scaling_factor = 5000
    rpos = np.array([int(x) for x in scaling_factor * priceChanges / latest_price])
    
    currentPos = np.array([int(x) for x in currentPos + rpos])
    return currentPos

