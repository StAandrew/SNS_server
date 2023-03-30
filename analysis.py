import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from rnn_model import get_prediction


# Price in day D
def get_price(ticker, day):
    prediction = get_prediction(ticker, day)['Close'].tolist()
    return prediction[day-1]

# Return over the next D days
def get_return(ticker, days):
    prediction = get_prediction(ticker, days)['Close'].tolist()
    profit = (prediction[days-1] - prediction[0]) / prediction[0]
    return profit

# Daily return over the next D days
def get_daily_returns(ticker, days):
    prediction = get_prediction(ticker, days)['Close'].tolist()
    daily_returns = []
    for i in range(days-1):
        daily_return = prediction[i+1] - prediction[i]
        daily_returns.append(daily_return)

    return daily_returns

# Average daily return over the next D days
def get_avg_daily_return(ticker, days):
    daily_returns = get_daily_returns(ticker, days)
    return np.mean(daily_returns)

# Volatility over the next D days
def get_std(ticker, days):
    prediction = get_prediction(ticker, days)['Close'].tolist()
    vol = np.std(prediction)
    return vol

# Sharpe ratio of stock over the next D days
def get_sharpe(ticker, days, rfr):
    avg_return = get_avg_daily_return(ticker, days)
    vol = get_std(ticker, days)
    sharpe = (avg_return - rfr) / vol
    return sharpe

# Get portfolio returns over the next D days
def get_portfolio_returns(ticker_list, days):
    returns_list = []
    for ticker in ticker_list:
        returns_list.append(get_daily_returns(ticker, days))

    combined_returns = np.array(returns_list).T

    return combined_returns

# Optimises portfolio of tickers for minimum variance
def min_var_portfolio(combined_returns):
    
    cov_matrix = np.cov(combined_returns.T)

    # Define the objective function
    def objective_function(weights, cov_matrix):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # Define the constraints
    def constraint_function(weights):
        return np.sum(weights) - 1.0
    
    # Define initial weights as equal weights
    n_assets = combined_returns.shape[1]
    init_weights = np.ones(n_assets) / n_assets
    
    # Define the bounds for the optimization
    bounds = tuple((0, 1) for i in range(n_assets))
    
    # Define the constraints for the optimization
    constraints = ({'type': 'eq', 'fun': constraint_function})
    
    # Use quadratic programming to minimize the variance
    result = minimize(objective_function, init_weights, args=cov_matrix, bounds=bounds, constraints=constraints, method='SLSQP')
    
    # Extract the optimal weights and minimum Variance achieved
    opt_weights = result.x
    min_var = result.fun

    return opt_weights, min_var

# Sharpe ratio of portfolio over the next D days
def get_portfolio_sharpe(weights, combined_returns, rfr):

    cov_matrix = np.cov(combined_returns.T)

    # Calculate portfolio returns and volatility
    total_return = np.sum(combined_returns.mean(axis=0) * weights)
    total_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    sharpe = (total_return - rfr) / total_vol

    return -sharpe      # negative return because we use it to minimise

# Optimises portfolio of tickers for maximum sharpe
def max_sharpe_portfolio(combined_returns, rfr):
    
    # Define the initial portfolio weights
    num_stocks = combined_returns.shape[1]
    init_weights = np.ones(num_stocks) / num_stocks

    # Define the optimization constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for i in range(num_stocks))

    # Run the optimization
    result = minimize(get_portfolio_sharpe, init_weights, args=(combined_returns, rfr), method='SLSQP', bounds=bounds, constraints=constraints)

    # Extract the optimal weights and maximum Sharpe ratio achieved
    opt_weights = result.x
    max_sharpe = -result.fun

    return opt_weights, max_sharpe



ticker_list = ['QQQ', 'AAPL', 'AMZN', 'IVV', 'URA']
combined_returns = get_portfolio_returns(ticker_list, 20)
opt_weights, max_sharpe = max_sharpe_portfolio(combined_returns, 0.01)
var_weights, min_var = min_var_portfolio(combined_returns)

print('--- SHARPE OPTIMISATION ---')
for i in range(5):
    print(f'{ticker_list[i]} weight: {opt_weights[i]}')
print(f'Max Sharpe: {max_sharpe}')
print(f'SUM CHECK: {sum(opt_weights)}')

print('---- VAR OPTIMISATION -----')
for i in range(5):
    print(f'{ticker_list[i]} weight: {var_weights[i]}')
print(f'Min Var: {min_var}')
print(f'SUM CHECK: {sum(var_weights)}')