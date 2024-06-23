import numpy as np
from joblib import Parallel, delayed
from numba import jit
import warnings

warnings.filterwarnings('ignore')

nInst = 50  # number of instruments (stocks)
currentPos = np.zeros(nInst)

def calculate_correlation_matrix(prcSoFar):
    return np.corrcoef(prcSoFar)

def select_top_pairs(correlation_matrix, top_n=10):
    pairs = []
    for i in range(correlation_matrix.shape[0]):
        for j in range(i + 1, correlation_matrix.shape[1]):
            pairs.append((i, j, abs(correlation_matrix[i, j])))
    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:top_n]
    return [(pair[0], pair[1]) for pair in pairs]

@jit(nopython=True)
def simple_moving_average(stock_series, window):
    return np.convolve(stock_series, np.ones(window)/window, mode='valid')

def generate_signals(prcSoFar):
    short_window = 5
    long_window = 20
    
    short_ma = np.zeros((nInst, prcSoFar.shape[1] - short_window + 1))
    long_ma = np.zeros((nInst, prcSoFar.shape[1] - long_window + 1))
    
    for i in range(nInst):
        short_ma[i] = simple_moving_average(prcSoFar[i], short_window)
        long_ma[i] = simple_moving_average(prcSoFar[i], long_window)
    
    signals = np.zeros(nInst)
    
    for i in range(nInst):
        if short_ma[i][-1] > long_ma[i][-1]:
            signals[i] = 1  # Buy signal
        elif short_ma[i][-1] < long_ma[i][-1]:
            signals[i] = -1  # Sell signal
    
    return signals

def getMyPosition(prcSoFar):
    global currentPos
    nInst, nt = prcSoFar.shape

    if nt < 20:
        return np.zeros(nInst)

    # Calculate the correlation matrix
    correlation_matrix = calculate_correlation_matrix(prcSoFar)

    # Select top pairs based on correlation
    top_pairs = select_top_pairs(correlation_matrix, top_n=10)
    
    # Generate signals using moving averages
    signals = generate_signals(prcSoFar)
    
    latest_price = prcSoFar[:, -1]

    # Calculate volatility for each instrument
    volatility = np.std(prcSoFar, axis=1, ddof=1)

    # Avoid division by zero and extremely low volatility
    volatility = np.where(volatility == 0, 1, volatility)

    # Adjust position based on signals
    rpos = signals * (10000 / latest_price) / 10  # More conservative scaling

    # Pairs trading adjustments
    for stock1, stock2 in top_pairs:
        if abs(latest_price[stock1] - latest_price[stock2]) > 2 * np.std(prcSoFar[stock1] - prcSoFar[stock2]):
            # Long the underperforming stock, short the outperforming stock
            if latest_price[stock1] < latest_price[stock2]:
                rpos[stock1] += 100
                rpos[stock2] -= 100
            else:
                rpos[stock1] -= 100
                rpos[stock2] += 100

    # Apply position limits
    max_positions = 10000 / latest_price
    rpos = np.clip(rpos, -max_positions, max_positions)
    
    new_positions = currentPos + rpos
    currentPos = np.clip(new_positions, -max_positions, max_positions).astype(int)

    return currentPos
