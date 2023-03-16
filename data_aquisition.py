import os
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from config import apple_dir


def get_apple_historical_data():

    """
    Retrieves historical stock data for Apple Inc. from Yahoo Finance.

    Args:
        None

    Returns:
        pandas.DataFrame:
            A DataFrame containing the historical stock data for Apple Inc.

    """
    print("Getting apple stock data...")
    # create a Ticker object for Apple Inc.
    apple = yf.Ticker("AAPL")

    # get the historical stock data for Apple Inc from 5 years ago until today.
    aapl_stock = apple.history(period="5y", interval="1d")

    # reset the index of the DataFrame
    aapl_stock.reset_index(inplace=True)

    # return the DataFrame containing the stock data
    return aapl_stock


def get_updated_stock_data(begin_date):
    """
    Retrieves the stock data for Apple Inc. from the previous trading day.

    Args:
        None

    Returns:
        pandas.DataFrame:
            A DataFrame containing the stock data for Apple Inc. from the previous
            trading day.

    """
    print("Getting updated apple stock data...")
    # create a Ticker object for Apple Inc.
    apple = yf.Ticker("AAPL")

    # get the stock data for Apple Inc. from begin_date to end_date
    aapl_stock = apple.history(start=begin_date, interval="1d")

    # reset the index of the DataFrame
    aapl_stock.reset_index(inplace=True)

    # remove first row since it is the same as the last row of the previous DataFrame
    aapl_stock = aapl_stock.iloc[1:, :]

    # return the DataFrame containing the stock data
    return aapl_stock


def process_apple_stock(apple):
    """
    Processes Apple stock data by removing unnecessary columns, calculating daily returns, and
    converting the datetime column to just the date. Also draws a heatmap correlation matrix
    between Apple stock features and saves the image.

    Args:
        apple: A Pandas DataFrame containing Apple stock data.

    Returns:
        A Pandas DataFrame containing the processed Apple stock data.

    Example Usage:
        apple_data = pd.read_csv('apple_stock_data.csv')
        processed_apple_data = process_apple_stock(apple_data)
    """
    print("Processing apple stock data...")
    # remove unnecessary columns
    apple = apple.drop(["Stock Splits"], axis=1)
    # add daily return column
    apple["Daily Return"] = apple["Close"].pct_change()
    # multiply daily return by 100 to get percentage
    apple["Daily Return"] = apple["Daily Return"].apply(lambda x: x * 100)
    # fill NaN values with 0
    apple["Daily Return"] = apple["Daily Return"].fillna(0)
    # convert datetime column to just date
    apple["Date"] = apple["Date"].apply(lambda x: x.date())
    return apple


def save_locally(data, directory):
    print("Saving data locally...")
    data.to_csv(directory)
    print("Data saved locally to: " + directory)
