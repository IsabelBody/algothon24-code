
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
