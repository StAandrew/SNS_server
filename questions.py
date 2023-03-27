import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from rnn_model import get_prediction


# Price in day D
def get_price(ticker, day):
    prediction = get_prediction(ticker, day)
    return prediction[day-1]

# Return over the next D days
def get_return(ticker, days):
    prediction = get_prediction(ticker, days)
    profit = (prediction[days-1] - prediction[0]) / prediction[0]
    return profit

# Daily return over the next D days
def get_daily_returns(ticker, days):
    prediction = get_prediction(ticker, days)
    daily_returns = []
    for i in range(days-1):
        daily_return = prediction[i+1] - prediction[i-1]
        daily_returns.append(daily_return)

    return daily_returns

# Average daily return over the next D days
def get_avg_daily_return(ticker, days):
    daily_returns = get_daily_returns(ticker, days)
    return np.mean(daily_returns)

# Volatility over the next D days
def get_vol(ticker, days):
    prediction = get_prediction(ticker, days)
    vol = np.std(prediction)
    return vol

# Sharpe ratio over the next D days
def get_sharpe(ticker, days):
    avg_return = get_avg_daily_return(ticker, days)
    vol = get_vol(ticker, days)
    sharpe = avg_return / vol
    return sharpe

# Stocks/bonds portfolio breakdown for optimising 
def stocks_or_bonds(days, metric):
    stocks = 'IVV'  # SP500 ETF
    bonds = 'TLT'   # Treasury ETF
    stock_return = get_return(stocks, days)
    bond_return = get_return(bonds, days)
    stock_pred = get_prediction(stocks, days)
    bond_pred = get_prediction(bonds, days)

    if metric == 'return':
        if get_return(stocks)