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
    # create a Ticker object for Apple Inc.
    apple = yf.Ticker("AAPL")

    # get the historical stock data for Apple Inc from 5 years ago until today.
    aapl_stock = apple.history(period="5y", interval="1d")

    # reset the index of the DataFrame
    aapl_stock.reset_index(inplace=True)

    # return the DataFrame containing the stock data
    return aapl_stock


def get_yesterdays_stock_data():
    """
    Retrieves the stock data for Apple Inc. from the previous trading day.

    Args:
        None

    Returns:
        pandas.DataFrame:
            A DataFrame containing the stock data for Apple Inc. from the previous
            trading day.

    """
    # create a Ticker object for Apple Inc.
    apple = yf.Ticker("AAPL")

    # get the stock data for Apple Inc. from the previous trading day
    aapl_stock = apple.history(period="1d", interval="1d")

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


# # apple = get_apple_data()
# # apple = process_apple_stock(apple)
# # save_locally(apple, apple_dir)

# # Get yesterday's price
# yesterday_data = get_yesterdays_stock_data()
# yesterday_data = process_apple_stock(yesterday_data)
# yesterday_data.set_index("Date", inplace=True)

# yesterday_price = yesterday_data.iloc[:,3:4].values[0][0]
# dataset = pd.read_csv(apple_dir, index_col=0)
# dataset.set_index("Date", inplace=True)
# # add yesterday's data to the dataset
# dataset_total = pd.concat((dataset, yesterday_data), axis = 0)
# trainset = dataset_total.iloc[-61:,3:4].values
# from sklearn.preprocessing import MinMaxScaler
# sc = MinMaxScaler(feature_range = (0, 1))
# import numpy as np
# training_scaled = sc.fit_transform(trainset)
# x_train =[]
# y_train =[]

# x_train.append(training_scaled[0:60, 0])
# y_train.append(training_scaled[60, 0])
# x_train, y_train = np.array(x_train),np.array(y_train)

# x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

# print(y_train)