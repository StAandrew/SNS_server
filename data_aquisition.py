import os
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from config import apple_dir


def get_apple_data():
    """
    Retrieves historical stock data for Apple Inc. from Yahoo Finance.

    Args:
        None

    Returns:
        pandas.DataFrame:
            A DataFrame containing the historical stock data for Apple Inc.

    """
    # create a Ticker object for Apple Inc.
    apple = yf.Ticker("AAPL")

    # get the historical stock data for Apple Inc.
    aapl_stock = apple.history(start="2017-03-01", end="2023-03-01", interval="1d")

    # reset the index of the DataFrame
    aapl_stock.reset_index(inplace=True)

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
    """
    Saves a Pandas DataFrame locally as a CSV file.

    Args:
        data: A Pandas DataFrame to be saved.
        dir: The directory where the CSV file should be saved.

    Returns:
        None

    Example Usage:
        stock_data = pd.read_csv('stock_data.csv')
        save_locally(stock_data, 'C:/Users/MyUser/Documents/stock_data.csv')
    """
    if not os.path.exists(directory):
        data.to_csv(directory)
        print("Data saved locally to: " + directory)
    else:
        print("Data already exists locally at: " + directory)


apple = get_apple_data()
apple = process_apple_stock(apple)
save_locally(apple, apple_dir)
