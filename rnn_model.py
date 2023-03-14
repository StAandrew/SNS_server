import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from data_aquisition import get_apple_historical_data, get_yesterdays_stock_data,process_apple_stock, save_locally
from config import apple_dir


apple = get_apple_data()
apple = process_apple_stock(apple)
save_locally(apple, apple_dir)
dataset = pd.read_csv(apple_dir, index_col=0)
dataset.set_index("Date", inplace=True)

# Train the model
def train_model():
    dataset = get_apple_historical_data()
    dataset = process_apple_stock(dataset)
    save_locally(dataset, apple_dir)
    dataset.set_index("Date", inplace=True)


    #train_data = dataset.loc["2017-03-01":"2023-01-01", :]
    #test_data = dataset.loc["2023-01-01":, :]
    #trainset = train_data.iloc[:,3:4].values

    trainset = dataset.iloc[:,3:4].values

    sc = MinMaxScaler(feature_range = (0,1))
    training_scaled = sc.fit_transform(trainset)

    x_train = []
    y_train = []
    for i in range(60,training_scaled.shape[0]):
        x_train.append(training_scaled[i-60:i, 0])
        y_train.append(training_scaled[i,0])
    x_train,y_train = np.array(x_train),np.array(y_train)

    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

    regressor = Sequential()

    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],1)))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    regressor.fit(x_train, y_train, epochs = 20, batch_size = 32)

    # Save the model
    regressor.save('stock_predictor.h5')

# Update the model with yesterday's price
def update_model():
    # Load the saved model
    regressor = load_model('stock_predictor.h5')

    # Get yesterday's price
    yesterday_data = get_yesterdays_stock_data()
    yesterday_data = process_apple_stock(yesterday_data)
    yesterday_data.set_index("Date", inplace=True)

    dataset = pd.read_csv(apple_dir, index_col=0)
    dataset.set_index("Date", inplace=True)
    # add yesterday's data to the dataset
    dataset_total = pd.concat((dataset, yesterday_data), axis = 0)
    trainset = dataset_total.iloc[-61:,3:4].values

    sc = MinMaxScaler(feature_range = (0,1))
    training_scaled = sc.fit_transform(trainset)
    x_train =[]
    y_train =[]
    x_train.append(training_scaled[0:60, 0])
    y_train.append(training_scaled[60, 0])
    x_train, y_train = np.array(x_train),np.array(y_train)

    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

    # Train the model
    regressor.fit(x_train, y_train, epochs = 1, batch_size = 32)

    # Get the predicted price
    inputs = dataset_total[-60:, 3:4].values
    predicted_price = regressor.predict(inputs)
    predicted_price = sc.inverse_transform(predicted_price)
    # Save the updated model
    regressor.save('stock_predictor.h5')
    return predicted_price

# Train the model if the saved model doesn't exist
try:
    regressor = load_model('stock_predictor.h5')
    prediction = update_model()
    print(prediction)
except:
    train_model()


def plot_prediction():
    plt.plot(testset, color = 'red', label = 'Real Apple Stock Price')
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Apple Stock Price')
    plt.title('Apple Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Apple Stock Price')
    plt.legend()
    plt.show()