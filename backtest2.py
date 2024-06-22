#!/usr/bin/env python


import numpy as np
import pandas as pd
from arimaposition import getMyPosition as getPosition



# In-sample mean(PL): 9.2
# In-sample return: 0.01311
# In-sample StdDev(PL): 55.92
# In-sample annSharpe(PL): 2.61
# In-sample totDvolume: 162092
# In-sample Score: 3.65

# Out-of-sample mean(PL): 20.9
# Out-of-sample return: 0.02316
# Out-of-sample StdDev(PL): 101.71
# Out-of-sample annSharpe(PL): 3.25
# Out-of-sample totDvolume: 207566
# Out-of-sample Score: 10.73


nInst = 0  # number of stocks
nt = 0  # number of days
commRate = 0.0010  # commission rate
dlrPosLimit = 10000  # limit per stock

# get data into a dataframe
def loadPrices(fn):  # (fn is filename)
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T  # transpose to get nInst x nt

pricesFile = "./prices.txt"
prcAll = loadPrices(pricesFile)
print("Loaded %d instruments for %d days" % (nInst, nt))

def calcPL(prcHist, start_day, end_day):
    """
    Calculate the PL (profit/loss) based on historical prices.
    """
    cash = 0  # current cash balance
    curPos = np.zeros(nInst)  # initiate positions (number of shares) as 0
    totDVolume = 0  # Total dollar volume traded
    value = 0  # current portfolio value in $
    todayPLL = []  # list to store today's profit and loss

    (_, nt) = prcHist.shape  # get number of days

    # Ensure we have enough data points for the model
    min_days_required = 20

    for t in range(start_day + min_days_required, end_day + 1):  # start after min_days_required
        prcHistSoFar = prcHist[:, :t]  # price history until current day
        newPosOrig = getPosition(prcHistSoFar)  # calculate position for day
        curPrices = prcHistSoFar[:, -1]  # get day's price from given data

        posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
        newPos = np.clip(newPosOrig, -posLimits, posLimits)
        deltaPos = newPos - curPos  # change in position from yesterday
        dvolumes = curPrices * np.abs(deltaPos)  # array of dollar volumes of trades for each stock
        dvolume = np.sum(dvolumes)  # total dollar volume of trades
        totDVolume += dvolume  # total volume of all days thus far
        comm = dvolume * commRate  # commission in $

        cash -= curPrices.dot(deltaPos) + comm  # update the cash balance
        curPos = np.array(newPos)  # update current positions
        posValue = curPos.dot(curPrices)  # calculate dollar value of portfolio
        todayPL = cash + posValue - value  # calculate profit (or loss)
        todayPLL.append(todayPL)  # add to array of all profit or loss calculations
        value = cash + posValue  # value of portfolio

        ret = 0.0
        if totDVolume > 0:
            ret = value / totDVolume

        print("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" %
              (t, value, todayPL, totDVolume, ret))

    pll = np.array(todayPLL)  # array of profit or loss from all days
    plmu, plstd = np.mean(pll), np.std(pll)
    annSharpe = 0.0
    if plstd > 0:
        annSharpe = np.sqrt(250) * plmu / plstd  # annualised Sharpe ratio assuming 250 days
    return plmu, ret, plstd, annSharpe, totDVolume

# Ensure minimum of 20 days for both periods
min_days_required = 20
total_days = prcAll.shape[1]
mid_point = total_days // 2

# Adjust the mid_point to ensure both samples have at least min_days_required
if mid_point < min_days_required:
    raise ValueError("Not enough total days to split the data with at least 20 days in each period.")

start_in_sample = 0
end_in_sample = mid_point - 1
start_out_sample = mid_point
end_out_sample = total_days - 1

# Ensure the out-sample period has at least min_days_required days
if end_out_sample - start_out_sample + 1 < min_days_required:
    start_out_sample = end_in_sample + 1
    if end_out_sample - start_out_sample + 1 < min_days_required:
        raise ValueError("Not enough total days to ensure out-sample period has at least 20 days.")

# In-sample period (training)
print("In-sample period:")
meanpl_in, ret_in, plstd_in, sharpe_in, dvol_in = calcPL(prcAll, start_in_sample, end_in_sample)
score_in = meanpl_in - 0.1 * plstd_in
print("=====")
print("In-sample mean(PL): %.1lf" % meanpl_in)
print("In-sample return: %.5lf" % ret_in)
print("In-sample StdDev(PL): %.2lf" % plstd_in)
print("In-sample annSharpe(PL): %.2lf" % sharpe_in)
print("In-sample totDvolume: %.0lf" % dvol_in)
print("In-sample Score: %.2lf" % score_in)

# Out-of-sample period (testing)
print("\nOut-of-sample period:")
meanpl_out, ret_out, plstd_out, sharpe_out, dvol_out = calcPL(prcAll, start_out_sample, end_out_sample)
score_out = meanpl_out - 0.1 * plstd_out
print("=====")
print("Out-of-sample mean(PL): %.1lf" % meanpl_out)
print("Out-of-sample return: %.5lf" % ret_out)
print("Out-of-sample StdDev(PL): %.2lf" % plstd_out)
print("Out-of-sample annSharpe(PL): %.2lf" % sharpe_out)
print("Out-of-sample totDvolume: %.0lf" % dvol_out)
print("Out-of-sample Score: %.2lf" % score_out)

