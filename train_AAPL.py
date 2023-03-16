import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from config import apple_dir

# Select dataset and split train and test
dataset = pd.read_csv(apple_dir, index_col=0)
dataset.set_index("Date", inplace=True)
train_data = dataset.loc["2017-03-01":"2023-01-01", :]
test_data = dataset.loc["2023-01-01":, :]

# Select closing price data
trainset = train_data.iloc[:,3:4].values

# Scaling
sc = MinMaxScaler(feature_range = (0,1))
training_scaled = sc.fit_transform(trainset)

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

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

# Defining loss function and optimiser
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Training model
regressor.fit(x_train, y_train, epochs = 20, batch_size = 32)

# Save model
regressor.save('Models/model_test')

# Load model
model = load_model('Models/model_test')

# Set up of the test set
testset = test_data.iloc[:,3:4].values

# Combine datasets and grab test inputs
dataset_total = pd.concat((train_data['Close'], test_data['Close']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test_data) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

# Take last 60 days
x_test = []

for i in range(60,inputs.shape[0]):
    x_test.append(inputs[i-60:i, 0])

x_test = np.array(x_test)

# Reshape inputs
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

# Predict price using trained model
predicted_stock_price = model.predict(x_test)
# Scale back to normal
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Plotting predictions vs actual
plt.plot(testset, color='red', label='Actual AAPL Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted AAPL Stock Price')
plt.title('Apple (AAPL) Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()