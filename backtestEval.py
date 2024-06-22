#!/usr/bin/env python

import numpy as np
import pandas as pd
from arimaposition import getMyPosition as getPosition

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

def calcPL(prcHist, start_day, end_day):
    """
    Calculate the PL (profit/loss) based on historical prices.
    """
    cash = 0 # current cash balance
    curPos = np.zeros(nInst) # initiate positions (number of shares) as 0
    totDVolume = 0
    value = 0 # current portfolio value in $
    todayPLL = [] # list to store today's profit and loss
    for t in range(start_day, end_day): # run from start_day to end_day
        prcHistSoFar = prcHist[:, :t] # price history until current day
        newPosOrig = getPosition(prcHistSoFar) # calculate position for day 
        curPrices = prcHistSoFar[:, -1] # get day's price from given data
        posLimits = np.array([int(x) for x in dlrPosLimit / curPrices]) # limit divided by yesterdays price 
        newPos = np.clip(newPosOrig, -posLimits, posLimits) # adjust new position (but within limit)
        deltaPos = newPos - curPos # change in position from yesterday
        dvolumes = curPrices * np.abs(deltaPos) # array of dollar volumes of trades for each stock
        dvolume = np.sum(dvolumes) # total dollar volume of trades
        totDVolume += dvolume # total volume of all days thus far
        comm = dvolume * commRate # commission in $
        cash -= curPrices.dot(deltaPos) + comm # update the cash balance in my trading account
        curPos = np.array(newPos)  # update global current positions with calculated new positions.
        posValue = curPos.dot(curPrices) # calculate dollar value of portfolio
        todayPL = cash + posValue - value # calculate profit (or loss)
        todayPLL.append(todayPL) # add to array of all profit or loss calculations
        value = cash + posValue # value of portfolio
        ret = 0.0
        if (totDVolume > 0):
            ret = value / totDVolume # calculate daily return (percentage)
        print("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" %
              (t, value, todayPL, totDVolume, ret))
    
    pll = np.array(todayPLL) # array of profit or loss from all days 
    (plmu, plstd) = (np.mean(pll), np.std(pll)) # mean and standard deviation of profit or loss
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(250) * plmu / plstd # annualised Sharpe ratio assuming 250 days 
    return (plmu, ret, plstd, annSharpe, totDVolume)

def backtest(prcAll, split_ratio=0.8):
    split_day = int(nt * split_ratio)
    
    # Split data into training and testing sets
    prcTrain = prcAll[:, :split_day]
    prcTest = prcAll[:, split_day+1:]
    
    # Evaluate on training set
    print("\nTraining Set Performance:")
    (meanpl_train, ret_train, plstd_train, sharpe_train, dvol_train) = calcPL(prcTrain, 250, split_day)
    score_train = meanpl_train - 0.1 * plstd_train
    print("=====")
    print("Training mean(PL): %.1lf" % meanpl_train)
    print("Training return: %.5lf" % ret_train)
    print("Training StdDev(PL): %.2lf" % plstd_train)
    print("Training annSharpe(PL): %.2lf" % sharpe_train)
    print("Training totDvolume: %.0lf" % dvol_train)
    print("Training Score: %.2lf" % score_train)

    # Evaluate on testing set
    print("\nTesting Set Performance:")
    (meanpl_test, ret_test, plstd_test, sharpe_test, dvol_test) = calcPL(prcTest, 0, prcTest.shape[1])
    score_test = meanpl_test - 0.1 * plstd_test
    print("=====")
    print("Testing mean(PL): %.1lf" % meanpl_test)
    print("Testing return: %.5lf" % ret_test)
    print("Testing StdDev(PL): %.2lf" % plstd_test)
    print("Testing annSharpe(PL): %.2lf" % sharpe_test)
    print("Testing totDvolume: %.0lf" % dvol_test)
    print("Testing Score: %.2lf" % score_test)

backtest(prcAll)
