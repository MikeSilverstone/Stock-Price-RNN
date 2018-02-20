import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.pyplot as plt


def import_manip_data_sets():
    dataset_train = pd.read_csv('Datasets/Recurrent_Neural_Networks/Google_Stock_Price_Train.csv')
    training_set = dataset_train.iloc[:, 1:2].values

    sc = MinMaxScaler(feature_range=(0, 1))
    normalized_training_data = sc.fit_transform(training_set)

    dataset_test = pd.read_csv('Datasets/Recurrent_Neural_Networks/Google_Stock_Price_Test.csv')
    real_stock_price = dataset_test.iloc[:, 1:2].values

    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1, 1)
    normalized_inputs = sc.transform(inputs)

    x_test = []
    for i in range(60, 80):
        x_test.append(normalized_inputs[i-60: i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return normalized_training_data, normalized_inputs, x_test, real_stock_price, sc


def create_rnn_model(shape_for_input):
    # Create model object
    regressor = Sequential()

    # Add first LSTM module
    regressor.add(LSTM(units=50,
                       return_sequences=True,
                       input_shape=(shape_for_input.shape[1], 1)))
    regressor.add(Dropout(rate=0.2))

    # Add second LSTM module
    regressor.add(LSTM(units=50,
                       return_sequences=True))
    regressor.add(Dropout(rate=0.2))

    # Add third LSTM module
    regressor.add(LSTM(units=50,
                       return_sequences=True))
    regressor.add(Dropout(rate=0.2))

    # Add fourth LSTM module
    regressor.add(LSTM(units=50,
                       return_sequences=False))
    regressor.add(Dropout(rate=0.2))

    # Add final layer
    regressor.add(Dense(units=1))

    # Compile module object
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    return regressor


def split_data():
    x_data = []
    y_data = []

    for i in range(60, len(normalized_training_set)):
        x_data.append(normalized_training_set[i - 60:i, 0])  # Previous 60 stocks from i
        y_data.append(normalized_training_set[i, 0])  # Current stock from i
    x_data, y_data = np.array(x_data), np.array(y_data)  # Convert to np array

    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))  # Features, time steps, indicators

    return x_data, y_data


def make_graph(real, predicted):
    plt.plot(real, color='red', label='Real Google Stock Price')
    plt.plot(predicted, color='blue', label='Predicted Google Stock Price')
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


# Define Hyper Parameters
HP_optimizer = ['adam', 'RMSprop', 'Nadam']
HP_dropout = [0.2, 0.3, 0.4]
HP_epochs = [100, 75, 50, 25]
HP_batch_size = [32, 16, 8]

# Fetch normalized training set
normalized_training_set, normalized_input_set, x_test_set, correct_stock_price, scaler = import_manip_data_sets()

# Fetch split x and y train datasets
x_train, y_train = split_data()

# Fetch model object
model = create_rnn_model(x_train)

# Fit training data to model
model.fit(x=x_train, y=y_train, epochs=HP_epochs[0], batch_size=HP_batch_size[0])

# Predict stock price
predicted_stock_price = model.predict(x_test_set)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Visualize results
make_graph(correct_stock_price, predicted_stock_price)
