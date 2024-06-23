import numpy as np
from scipy import stats
from joblib import Parallel, delayed

nInst = 50  # number of instruments (stocks)
currentPos = np.zeros(nInst)

def find_cointegrated_pairs(prices):
    pairs = []
    n = prices.shape[0]
    # Calculate Pearson correlation coefficients
    correlations = np.corrcoef(prices)
    
    # Find pairs with high correlation (adjust threshold as needed)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(correlations[i, j]) > 0.8:  # Example threshold for high correlation
                pairs.append((i, j))
    
    return pairs

def calculate_spread(prices1, prices2):
    # Calculate simple spread (difference) between two price series
    spread = prices1 - prices2
    return spread

def generate_signals(spread):
    # Generate signals based on z-score of the spread
    zscore = (spread - np.mean(spread)) / np.std(spread)
    long_signal = zscore < -1
    short_signal = zscore > 1
    return long_signal, short_signal

def process_pair(pair, prcSoFar):
    i, j = pair
    spread = calculate_spread(prcSoFar[i], prcSoFar[j])
    long_signal, short_signal = generate_signals(spread)

    pos_i = 0
    pos_j = 0

    if long_signal[-1]:  # Go long on the spread
        pos_i = 1000  # Long stock i
        pos_j = -1000  # Short stock j
    elif short_signal[-1]:  # Go short on the spread
        pos_i = -1000  # Short stock i
        pos_j = 1000  # Long stock j

    return i, pos_i, j, pos_j

def getMyPosition(prcSoFar):
    global currentPos
    nInst, nt = prcSoFar.shape

    if nt < 2:
        return np.zeros(nInst)
    
    # Reduce the number of pairs based on a heuristic or pre-filtering
    pairs = find_cointegrated_pairs(prcSoFar)
    pairs = pairs[:100]  # Example: Limit to the first 100 pairs
    
    new_positions = np.zeros(nInst)
    
    # Parallel processing for pair trading
    results = Parallel(n_jobs=-1)(delayed(process_pair)(pair, prcSoFar) for pair in pairs)
    
    for i, pos_i, j, pos_j in results:
        new_positions[i] += pos_i
        new_positions[j] += pos_j

    # Apply position limits
    latest_price = prcSoFar[:, -1]
    max_positions = 10000 / latest_price
    new_positions = np.clip(new_positions, -max_positions, max_positions)
    
    currentPos = np.clip(currentPos + new_positions, -max_positions, max_positions).astype(int)
    return currentPos
