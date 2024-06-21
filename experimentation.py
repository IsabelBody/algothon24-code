import numpy as np
from sklearn.linear_model import LinearRegression

# Define the getMyPosition function
def getMyPosition(prices):
    nInst, nt = prices.shape
    positions = np.zeros(nInst)
    
    # Parameters
    trend_threshold = 0.05  # Threshold for trend detection (slope)
    volatility_threshold = 0.02  # Threshold for volatility detection (std deviation)
    
    # Calculate returns
    returns = np.diff(prices, axis=1) / prices[:, :-1]
    
    for i in range(nInst):
        # Get the price series for the current stock
        price_series = prices[i, :]
        
        # Fit linear regression to detect trend
        X = np.arange(nt).reshape(-1, 1)
        y = price_series
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        
        # Calculate volatility
        volatility = np.std(returns[i, :])
        
        # Apply different strategies based on trend and volatility
        if abs(slope) > trend_threshold:
            # Trending stock: Momentum strategy
            if slope > 0:
                # Uptrend: Go long
                positions[i] = 100
            else:
                # Downtrend: Go short
                positions[i] = -100
        elif volatility > volatility_threshold:
            # Volatile stock: Mean-reversion strategy
            mean_price = np.mean(price_series)
            if price_series[-1] < mean_price:
                # Price is below mean: Buy
                positions[i] = 100
            else:
                # Price is above mean: Sell
                positions[i] = -100
        else:
            # Neutral or low volatility: Hold position
            positions[i] = 0
    
    return positions
