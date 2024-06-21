#!/usr/bin/env python

import numpy as np
import pandas as pd
from ema import getMyPosition as getPosition

nInst = 0 # number of stocks 
nt = 0 # number of days 
commRate = 0.0010 # commission rate
dlrPosLimit = 10000 # limit per stock


# get data into a dataframe
def loadPrices(fn): # (fn is filename)
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T # transpose to get nInst x nt


pricesFile = "./prices.txt"
prcAll = loadPrices(pricesFile)
print("Loaded %d instruments for %d days" % (nInst, nt))


def calcPL(prcHist):
    """
    Calculate the PL (profit/loss) based on historical prices.
    ---- not prfit & loss, right?

    """
    
    cash = 0 # current cash balance
    
    # initiate positions (number of shares) as 0
    curPos = np.zeros(nInst) 
    
    # Total dollar volume traded 
    totDVolume = 0
    totDVolumeSignal = 0
    totDVolumeRandom = 0
    value = 0 # current portfolio value in $
    todayPLL = [] # list to store today's profit and loss
    (_, nt) = prcHist.shape # get number of days 
    for t in range(250, 501): # run from day 250 to 500
        prcHistSoFar = prcHist[:, :t] # price history until current day
        newPosOrig = getPosition(prcHistSoFar) # calculate position for day 
        curPrices = prcHistSoFar[:, -1] # get day's price from given data
        
        # total number of positions we are able to buy or sell 
        # limit divided by yesterdays price 
        # make int because we need to hold whole shares
        posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
        
        # adjust new position (but within limit)
        newPos = np.clip(newPosOrig, -posLimits, posLimits)
        
        deltaPos = newPos - curPos # change in position from yesterday
        
        # array of dollar volumes of trades for each stock
        dvolumes = curPrices * np.abs(deltaPos) 
        # total dollar volume of trades
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume # total volume of all days thus far
        comm = dvolume * commRate # commission in $
        
         
        """ 
        update the cash balance in my trading account
        current cash - costs (cost of buying more shares, commission)
        
        current prices * change = the cost of buying more or the 
                                   revenue of selling some 
        + take off commission
        
        dot = dot product of two arrays
        
        (if we are selling, we gain money through the double negative)
        """
        cash -= curPrices.dot(deltaPos) + comm 
        # update global current positions with calculated new positions.
        curPos = np.array(newPos)  
        # calculate dollar value of portfolio
        posValue = curPos.dot(curPrices)
        # calculate profit (or loss)
        todayPL = cash + posValue - value
        # add to array of all profit or loss calculations
        todayPLL.append(todayPL) 
        value = cash + posValue # value of portfolio
        
        # calculate daily return (percentage)
        ret = 0.0
        if (totDVolume > 0):
            # value of portfolio today / total dollar volume across all days 
            ret = value / totDVolume
            
        # print results
        print("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" %
              (t, value, todayPL, totDVolume, ret))
    
    pll = np.array(todayPLL) # array of profit or loss from all days 
    
    # mean and standard deviation of profit or loss
    (plmu, plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        # annualised Sharpe ratio assuming 250 days 
        annSharpe = np.sqrt(250) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume)


(meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll)
score = meanpl - 0.1*plstd
print("=====")
print("mean(PL): %.1lf" % meanpl) # mean profit or loss
print("return: %.5lf" % ret) # return
print("StdDev(PL): %.2lf" % plstd) # standard deviation 


''' 
annualised Sharpe ratio: risk adjusted return 
tells us how much the portfolio is expected to excess in
return (aka reward, yay) after variability is applied to 
our mean through standard deviation.

Interpreting: 
The higher the Sharpe ration (risk-adjusted return), 
the better the portfolio.

A negative sharpe ratio indicates that the portfolio is 
worse than random chance.

Aim to get a Sharpe ration > 1
>1.0 is good
>2.0 is very good
>3.0 is excellent

more than 5 could be an overfitting issue

Don't just focus on the Sharpe ratio, backtest algo across different
data scenarios. 
'''

print("annSharpe(PL): %.2lf " % sharpe) 
print("totDvolume: %.0lf " % dvol)
print("Score: %.2lf" % score) # bigger the better
