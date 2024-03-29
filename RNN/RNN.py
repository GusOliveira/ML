#This RNN will predict the stock prize trend one month ahead.

#libraries
import numpy as np #numpy allows the use of arrays and dataframes
import matplotlib.pyplot as plt #matplotlib to visualize the results in graphs and plots
import pandas as pd #allow to import datasets
from sklearn.preprocessing import MinMaxScaler #normalizing the data between 1 and 0
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


#importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values


#feature scaling
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)


#create a data structure with 60 time steps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train) #create a np array    


#reshaping the data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#Build the RNN

#initialize the RNN
regressor = Sequential()

#add the first layer
regressor.add(LSTM(units=50, return_sequences= True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0,2))

#add the second layer
regressor.add(LSTM(units=50, return_sequences= True))
regressor.add(Dropout(0,2))

#add the third layer
regressor.add(LSTM(units=50, return_sequences= True))
regressor.add(Dropout(0,2))

#add the fourth layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0,2))

#add the output layer
regressor.add(Dense(units=1))


#Compile the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

#Fitting the RNN to the training Set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)


#Predict the results
#Get the test data
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#get the predict stock price
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


#Visualize the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.show()