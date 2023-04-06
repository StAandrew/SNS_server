import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize
from residual_model import get_prediction

from data_acquisition import get_historical_data, process_stock_data
import matplotlib.dates as mdates


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
    out = get_prediction(ticker, days)
    values = out['Close'].tolist()
    daily_returns = []
    for i in range(days-1):
        daily_return = values[i+1] - values[i]
        daily_returns.append(daily_return)

    out = out.tail(days-1)
    out['Close'] = out['Close'].replace(values[1:], daily_returns)
    return out

# Average daily return over the next D days
def get_avg_daily_return(ticker, days):
    daily_returns = get_daily_returns(ticker, days)['Close'].tolist()
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
        daily_returns = get_daily_returns(ticker, days)
        values = daily_returns['Close'].tolist()
        returns_list.append(values)
        
    dates = daily_returns.index.values

    combined_returns = np.array(returns_list).T
    combined_returns_df = pd.DataFrame(combined_returns, columns=ticker_list)
    combined_returns_df['Date'] = dates
    combined_returns_df.set_index('Date', inplace=True)

    return combined_returns, combined_returns_df

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



### SERVER-SIDE PLOT DEBUGGING - DELETE BEFORE SUBMIT ###


# def plot_prediction(ticker, predictions):
#     dataset = get_historical_data(ticker)
#     dataset = process_stock_data(dataset)
#     dataset.set_index("Date", inplace=True)
    
#     # Convert the index to datetime objects
#     dataset.index = pd.to_datetime(dataset.index)
#     predictions.index = pd.to_datetime(predictions.index)

#     fig, ax = plt.subplots()
    
#     ax.plot(dataset.tail(60).index, dataset.tail(60)['Close'], color='red', label=f'Historical Price')
#     ax.plot(predictions.index, predictions['Close'], color='blue', label=f'Predicted Price')
#     ax.set_title(f'{ticker} Price Prediction')
#     ax.set_xlabel('Time')
#     ax.set_ylabel(f'{ticker} Price')
#     plt.legend(loc='best')
#     plt.grid()
    
#     # Set the x-axis to display one label per month
#     ax.xaxis.set_major_locator(mdates.MonthLocator())
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
#     # Rotate the x-axis labels for better readability
#     plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
#     plt.savefig("Plots/test_prediction.png")
    
#     ax.legend()

# def plot_daily_returns(ticker, predictions):
    
#     # Convert the index to datetime objects
#     predictions.index = pd.to_datetime(predictions.index)

#     fig, ax = plt.subplots()
    
#     ax.plot(predictions.index, predictions['Close'], color='blue')
#     plt.axhline(y=0, color='k', linestyle='-')
#     ax.set_title(f'{ticker} Daily Return Prediction')
#     ax.set_xlabel('Time')
#     ax.set_ylabel(f'{ticker} Daily Return')
#     plt.grid()
    
#     # Set the x-axis to display one label per month
#     ax.xaxis.set_major_locator(mdates.MonthLocator())
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
#     # Rotate the x-axis labels for better readability
#     plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
#     plt.savefig("Plots/test_daily_ret.png")
    
#     ax.legend()

# def plot_daily_portfolio_returns(ticker_list, combined_returns):    # feed combined returns df

#     combined_returns.index = pd.to_datetime(combined_returns.index)
#     total_returns = combined_returns.sum(axis=1).reset_index(name='Total')

#     fig, ax = plt.subplots()

#     for ticker in ticker_list:
#         ax.plot(combined_returns.index, combined_returns[ticker], label=f'{ticker}')
#     ax.plot(combined_returns.index, total_returns['Total'], linewidth=2, label=f'Portfolio')
#     plt.axhline(y=0, color='k', linestyle='-')
#     ax.set_title(f'Portfolio Daily Return Prediction')
#     ax.set_xlabel('Time')
#     ax.set_ylabel(f'Daily Returns')
#     plt.legend(loc='best')
#     plt.grid()

#     # Set the x-axis to display one label per month
#     ax.xaxis.set_major_locator(mdates.MonthLocator())
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
#     # Rotate the x-axis labels for better readability
#     plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
#     plt.savefig("Plots/test_daily_port_ret.png")
    
#     ax.legend()

# def plot_cumulative_portfolio_returns(ticker_list, combined_returns):

#     combined_returns.index = pd.to_datetime(combined_returns.index)
#     cumulative_returns = combined_returns.cumsum()
#     total_returns = cumulative_returns.sum(axis=1).reset_index(name='Total')

#     fig, ax = plt.subplots()

#     for ticker in ticker_list:
#         ax.plot(cumulative_returns.index, cumulative_returns[ticker], label=f'{ticker}')
#     ax.plot(cumulative_returns.index, total_returns['Total'], linewidth=2, label=f'Portfolio')
#     plt.axhline(y=0, color='k', linestyle='-')
#     ax.set_title(f'Portfolio Cumulative Return Prediction')
#     ax.set_xlabel('Time')
#     ax.set_ylabel(f'Returns')
#     plt.legend(loc='best')
#     plt.grid()
    
#     # Set the x-axis to display one label per month
#     ax.xaxis.set_major_locator(mdates.MonthLocator())
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
#     # Rotate the x-axis labels for better readability
#     plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
#     plt.savefig("Plots/test_cum_port_ret.png")
    
#     ax.legend()

# def plot_opt_portfolio(ticker_list, weights, type):

#     fig, ax = plt.subplots()
#     ax.pie(weights, labels=ticker_list, autopct='%1.1f%%')
#     if type == 'var':
#         ax.set_title(f'Minimum Variance Portfolio')
#         plt.savefig("Plots/min_var_test.png")
#     elif type == 'sharpe':
#         ax.set_title(f'Maximum Sharpe Portfolio')
#         plt.savefig("Plots/test_max_sharpe.png")



# tkr = 'NVDA'
# tkr_list = ['AAPL', 'AMZN', 'NVDA', 'MSFT']
# days = 30

# out, out_df = get_portfolio_returns(tkr_list, days)
# print(out_df)
# plot_daily_portfolio_returns(tkr_list, out_df)

# # print(out)
# # out2, var = min_var_portfolio(out)
# # out3, sharpe = max_sharpe_portfolio(out, 0.02)
# # plot_opt_portfolio(tkr_list, out2, 'var')
# # plot_opt_portfolio(tkr_list, out3, 'sharpe')

# # out, out_df = get_portfolio_returns(tkr_list, days)
# # print(out_df)
# # plot_cumulative_portfolio_returns(tkr_list, out_df)