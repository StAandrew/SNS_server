import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from data_aquisition import get_apple_historical_data, get_updated_stock_data,process_apple_stock, save_locally
import os
import pickle
from config import apple_dir, figures_dir

import pandas as pd
from pandas_market_calendars import get_calendar

def next_market_date(date_str):
    date = pd.to_datetime(date_str)
    nyse = get_calendar('NYSE')
    end_date = pd.Timestamp(date.year+1, 12, 31)
    schedule = nyse.schedule(start_date=date, end_date=end_date)
    return schedule.iloc[1]['market_open'].strftime('%Y-%m-%d')

# Train the model
def train_model():
    print("Training model...")
    dataset = get_apple_historical_data()
    dataset = process_apple_stock(dataset)
    dataset.set_index("Date", inplace=True)
    save_locally(dataset, apple_dir)

    train_data = dataset.loc[:, ['Close']] # extract the closing price column
    # Scaling
    sc = MinMaxScaler(feature_range = (0,1))
    training_scaled = sc.fit_transform(train_data)
    
    pickle.dump(sc, open("scaler.pkl", "wb"))

    # Split train data between x and y components
    x_train = []
    y_train = []

    for i in range(60,training_scaled.shape[0]):
        x_train.append(training_scaled[i-60:i, 0])
        y_train.append(training_scaled[i,0])

    x_train,y_train = np.array(x_train),np.array(y_train)

    # Reshape inputs
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

    # Arranging keras layers in sequential order
    regressor = Sequential()
    # Layer setup
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    regressor.fit(x_train, y_train, epochs = 20, batch_size = 32)

    # Save the model
    regressor.save('stock_predictor.h5')

# Update the model with yesterday's price
def update_model(last_date):
    print("Updating model...")
    # Load the saved model
    regressor = load_model('stock_predictor.h5')

    # Get updated data
    new_data = get_updated_stock_data(last_date)

    print(new_data.tail())
    new_data = process_apple_stock(new_data)
    new_data.set_index("Date", inplace=True)

    dataset = pd.read_csv(apple_dir, index_col=0)

    # add new data to the dataset
    dataset_total = pd.concat((dataset, new_data), axis = 0)
    # save the updated dataset
    save_locally(dataset_total, apple_dir)

    # extract the closing price column
    dataset_total = dataset_total.loc[:, ['Close']] 
    
    sc = MinMaxScaler(feature_range = (0,1))
    sc = sc.fit(dataset_total)

    training_scaled = sc.transform(dataset_total)
    pickle.dump(sc, open("scaler.pkl", "wb"))
    x_train =[]
    y_train =[]
    print(new_data.shape[0])
    for i in range(new_data.shape[0]):
        if i == 0:
            x_train.append(training_scaled[-60:, 0])
            y_train.append(training_scaled[-1, 0])
        else:
            x_train.append(training_scaled[-i-60:-i, 0])
            y_train.append(training_scaled[-i-1, 0])
    x_train.reverse()
    y_train.reverse()
    x_train, y_train = np.array(x_train),np.array(y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

    # Train the model
    regressor.fit(x_train, y_train, epochs = 1, batch_size = 1)
    # Save the updated model
    regressor.save('stock_predictor.h5')

def predict(days):
    print("Predicting...")
    # Load the saved model
    regressor = load_model('stock_predictor.h5')
    dataset = pd.read_csv(apple_dir, index_col=0)
    # load the scaler
    sc = pickle.load(open("scaler.pkl", "rb"))

    prediced_close_prices = []
    dates = []
    for x in range(days):
        inputs = dataset.tail(60)['Close']
        inputs = inputs.values.reshape(-1,1)

        inputs = sc.transform(inputs)
        inputs = np.array(inputs[:,0])
        inputs = np.reshape(inputs, (1,inputs.shape[0],1))
        predicted_stock_price = regressor.predict(inputs)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        # append the predicted value to the train data with the next date generated by next_market_date
        last_date = dataset.index[-1]
        next_date = next_market_date(last_date)
        dataset.loc[next_date, 'Close'] = predicted_stock_price[0][0]
        prediced_close_prices.append(predicted_stock_price[0][0])
        dates.append(next_date)
        
    predictions = pd.DataFrame({'Date': dates, 'Close': prediced_close_prices})
    predictions.set_index('Date', inplace=True)
    return predictions

def plot_prediction_vs_real(real_stock_prices, predictions):
    plt.plot(real_stock_prices, color = 'red', label = 'Real Apple Stock Price')
    plt.plot(predictions, color = 'blue', label = 'Predicted Apple Stock Price')
    plt.title('Apple Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Apple Stock Price')
    plt.legend()
    plt.show()

def plot_prediction(predictions):
    dataset = pd.read_csv(apple_dir, index_col=0)
    plt.plot(dataset.index,dataset['Close'], color = 'red', label = 'Historical Apple Stock Price')
    plt.plot(predictions.index,predictions['Close'], color = 'blue', label = 'Predicted Apple Stock Price')
    plt.title('Apple Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Apple Stock Price')
    plt.legend()
    plt.savefig(os.path.join(figures_dir, "prediction.png"))
    plt.show()

#check for stock_predictor.h5
if not os.path.exists('stock_predictor.h5') or not os.path.exists(apple_dir):
    train_model()

data = pd.read_csv(apple_dir, index_col=0)
last_date = data.index[-1]
# check if the last row is today's date
if last_date != f"{pd.Timestamp.today().date()}":
    update_model(last_date)

prediction = predict(10)
print(prediction)
plot_prediction(prediction)