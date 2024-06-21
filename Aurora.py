
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50 # number of instruments (stocks) is 50

# initialise current position to 0 for each instrument
# we hold no shares, we've borrowed no shares
currentPos = np.zeros(nInst)



def predict_nextday_prices(prices):
    """
    Calculate the Exponential Moving Average (EMA). EMA can be my baseline. 
    
    """
    
    window = 10
    ema = np.zeros_like(prices)
    alpha = 2 / (window + 1)
    ema[0] = prices[0]
    
    for t in range(1, len(prices)):
        ema[t] = alpha * prices[t] + (1 - alpha) * ema[t - 1]
    
    return ema


def getMyPosition(prcSoFar):
    global currentPos
    
    # number of instruments, number of days
    # 50, n
    (nins, nt) = prcSoFar.shape
    
    ''' if it is day 1 
    return 0 for all positions for all 50 instruments
    this is because there is insufficient data 
    '''
    if (nt < 2):
        return np.zeros(nins) # exiting the function early 

    
    # initialise predicted prices as 0. 
    predictedPrices = np.zeros(nins) 
    
    # set predicted prices for each instrument (stock)
    for i in range(nins):
        ma = predict_nextday_prices(prcSoFar[i])
        predictedPrices[i] = ma[-1]
    
    
    latest_price = prcSoFar[:, -1]
    
    priceChanges = predictedPrices - latest_price
    
    # calculate norm of the data then normalise
    lNorm = np.sqrt(priceChanges.dot(priceChanges))
    priceChanges /= lNorm
    
    
    '''
    after I have a good algorithm for price estimation I should optimise for the best scaling factor 
    
    Volatility of the Instruments: Higher volatility might require a smaller scaling factor to avoid excessive trading and risk.
    Risk Tolerance: If you want a more conservative strategy, you might reduce the scaling factor.  
    Performance Optimization: You can run backtests with different values to see which one gives the best performance 
    according to your evaluation metrics (mean(PL) - 0.1 * StdDev(PL)).
    '''
    scaling_factor = 5000
    
    '''
    Reposition: convert changes of price into changes of position
    
    e.g +10 (buy 10 more)
    
    scaling_factor * priceChanges: This scales the prediction by a 
    factor. The factor is arbitrary and can be tuned. 
    It influences the magnitude of the positions taken.
    
    int(x) for x: converts it to integer as it is not possible to 
    take fractional shares.
    '''
    rpos = np.array([int(x) for x in scaling_factor * priceChanges / latest_price])

    
    #  updates positions with changes 
    currentPos = np.array([int(x) for x in currentPos+rpos])
    return currentPos



nInst = 50 # number of instruments (stocks) is 50

# initialise current position to 0 for each instrument
currentPos = np.zeros(nInst)

def make_stationary(series):
    """
    Apply differencing to make the series stationary.
    """
    return np.diff(series)

def predict_nextday_prices(prices):
    """
    Predict the next day's price for each instrument using ARIMA model.
    """
    predicted_prices = []
    for price_series in prices:
        try:
            # Apply differencing to make the series stationary
            stationary_series = make_stationary(price_series)
            
            # Fit ARIMA model: order (p,d,q) needs to be tuned for better results
            model = ARIMA(stationary_series, order=(5,1,0))
            model_fit = model.fit()
            
            # Forecast the next day price (difference)
            forecast_diff = model_fit.forecast(steps=1)[0]
            
            # Convert the forecasted difference back to the original scale
            forecast = price_series[-1] + forecast_diff
        except Exception as e:
            print(f"ARIMA fitting error: {e}")
            forecast = price_series[-1]  # Use the last price if the model fails
        predicted_prices.append(forecast)
    
    return np.array(predicted_prices)

def getMyPosition(prcSoFar):
    global currentPos
    
    # number of instruments, number of days
    (nins, nt) = prcSoFar.shape
    
    # Return zero positions if there is insufficient data
    if (nt < 2):
        return np.zeros(nins)
    
    # Predict the next day prices using ARIMA
    predictedPrices = predict_nextday_prices(prcSoFar)
    latest_price = prcSoFar[:, -1]
    
    priceChanges = predictedPrices - latest_price
    
    # Calculate norm of the data then normalize
    lNorm = np.sqrt(priceChanges.dot(priceChanges))
    priceChanges /= lNorm
    
    scaling_factor = 5000
    
    # Reposition: convert changes of price into changes of position
    rpos = np.array([int(x) for x in scaling_factor * priceChanges / latest_price])
    
    # Update positions with changes
    currentPos = np.array([int(x) for x in currentPos+rpos])
    return currentPos

