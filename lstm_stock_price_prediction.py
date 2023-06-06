import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Fetch the stock price data
stock_symbol = "GOOGL"  # tesla's stock symbol
start_date = "2015-01-01"
end_date = "2023-05-31"
data = yf.download(stock_symbol, start=start_date, end=end_date, progress=False)

# Step 2: Preprocess the data
data = data[['Close']]  # selecting only the 'Close' column
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

# Step 3: Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Step 4: Prepare the data for LSTM training
def create_dataset(dataset, time_steps=1):
    X, Y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:i + time_steps, 0])
        Y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(Y)

time_steps = 60  # number of previous time steps to use for prediction
X_train, Y_train = create_dataset(train_data, time_steps)
X_test, Y_test = create_dataset(test_data, time_steps)

# reshape the input data for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Step 5: Build and train the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=20, batch_size=32)

# Step 6: Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Step 7: Inverse scaling of predictions
train_predictions = scaler.inverse_transform(train_predictions).flatten()
test_predictions = scaler.inverse_transform(test_predictions).flatten()

# Step 8: Create a combined index for plotting
train_index = data.index[time_steps:train_size]
test_index = data.index[train_size + time_steps:]

# Step 9: Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data.index[:train_size], data['Close'][:train_size], label='Actual (Training)', color='blue')
plt.plot(data.index[train_size:], data['Close'][train_size:], label='Actual (Testing)', color='orange')
plt.plot(train_index, train_predictions, label='Predicted (Training)', color='red')
plt.plot(test_index, test_predictions, label='Predicted (Testing)', color='green')

plt.legend()

plt.title('GOOGL Actual vs. Predicted Stock Price')
plt.xlabel('Time')
plt.ylabel('Price ($)')

plt.show()