import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy.stats import norm

nInst = 50  # number of instruments (stocks)
currentPos = np.zeros(nInst)

# Function to calculate historical volatility
def historical_volatility(prices):
    returns = np.diff(prices) / prices[:-1]
    volatility = np.std(returns)
    return volatility

# Function to calculate Value at Risk (VaR)
def calculate_var(prices, confidence_level=0.95):
    returns = np.diff(prices) / prices[:-1]
    return norm.ppf(1 - confidence_level, loc=np.mean(returns), scale=np.std(returns))

# Function to determine best transformation based on ADF test
def determine_best_transformation(stock_series):
    result = adfuller(stock_series)
    if result[1] < 0.05:
        return 'Original'
    else:
        return 'Differencing'

# Function to fit ARIMA model and predict next day's price change
def fit_arima_model(stock_series, order=(1, 0, 0)):
    try:
        model = ARIMA(stock_series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)[0]
        return forecast
    except:
        return np.nan

# Main function to get positions based on current price data
def getMyPosition(prcSoFar):
    global currentPos
    nInst, nt = prcSoFar.shape

    if nt < 2:
        return np.zeros(nInst)
    
    # Calculate historical volatilities
    volatilities = [historical_volatility(prcSoFar[i, :]) for i in range(nInst)]
    max_volatility = np.max(volatilities)
    
    # Calculate Value at Risk (VaR) for each instrument
    VaRs = [calculate_var(prcSoFar[i, :]) for i in range(nInst)]
    
    # Initialize position changes
    position_changes = np.zeros(nInst)
    
    for i in range(nInst):
        stock_prices = prcSoFar[i]
        
        # Determine best transformation based on ADF test
        transformation = determine_best_transformation(stock_prices)
        
        # Apply transformation
        transformed_series = np.diff(stock_prices) if transformation == 'Differencing' else stock_prices
        
        # Fit ARIMA model
        predicted_price_change = fit_arima_model(transformed_series)
        
        if np.isnan(predicted_price_change):
            continue
        
        # Calculate risk-adjusted position change
        risk_adjusted_change = predicted_price_change / max_volatility
        
        # Scale position change based on VaR
        if VaRs[i] > 0:
            risk_adjusted_change *= VaRs[i]
        
        position_changes[i] = risk_adjusted_change
    
    # Scale position changes based on current positions and market conditions
    new_positions = currentPos + position_changes
    
    # Clip positions to $10k limit per stock
    max_positions = 10000 / prcSoFar[:, -1]
    new_positions = np.clip(new_positions, -max_positions, max_positions)
    
    # Update current positions and return
    currentPos = new_positions.astype(int)
    
    return currentPos
