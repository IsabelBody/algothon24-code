
import numpy as np

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50 # number of instruments (stocks) is 50

# initialise current position to 0 for each instrument
# we hold no shares, we've borrowed no shares
currentPos = np.zeros(nInst)


def getMyPosition(prcSoFar):
    # This should be a function which calls many other functions imo. 
    
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
    
    
    ''' 
     calculate log returns
     
     returns = PL(profit or loss)
     
     prcSoFar[:, -1] = today's price
     prcSoFar[:, -2] = yesterday's price
     
     prcSoFar[:, -1] / prcSoFar[:, -2] is the ratio 
     change from yesterday to today
     
     We used log as log returns are more normally distributed, 
     and reflect additive standarised change.
     
    '''
     # CHANGE lastRet to predicted_prices - last day
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])
    
    # calculate norm of the data 
    lNorm = np.sqrt(lastRet.dot(lastRet))
    # normalise to ensure magnitude of returns is normalised 
    lastRet /= lNorm
    
    ''' Reposition
    calculate position changes from returns into array
    
    5000 * lastRet: This scales the normalized log returns by a 
    factor of 5000. The factor of 5000 is arbitrary and can be tuned. 
    It influences the magnitude of the positions taken.
    And instead of lastRet, this is exactly where we should insert 
    our predicted_prices from our modelling for the next day.   
    
    
    int(x) for x: converts it to integer as it is not possible to 
    take fractional shares.
    
    '''
    rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])
    
    '''  updates positions with changes '''
    currentPos = np.array([int(x) for x in currentPos+rpos])
    return currentPos
