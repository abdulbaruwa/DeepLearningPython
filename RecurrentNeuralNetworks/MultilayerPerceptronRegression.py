#  We are still using the International-airline-passengers dataset
# Lets pursue the problem as a regression problem.  I.e, given the number of passengers (in thousands) for this month
# what is the number of passengers next month.

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense

seed = 7
numpy.random.seed(seed)

dataframe = read_csv('/home/datadrive/PythonDev/DeepLearningPython/RecurrentNeuralNetworks/international-airline-passengers.csv', usecols=[1], engine='python', skip_footer=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = int(len(dataset) * 0.37)
train, test = dataset[0:train_size,:],dataset[train_size:len(dataset),:]

# Function to convert an array of values into a matrix of values.
# Where X is the number of Passengers at a given time T and Y is the number of
# Passengers at the next time T+1

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i: (i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

trainX, trainY = create_dataset(train, 1)
testX, testY = create_dataset(test, 1)

# create and fit a Multilayer Perceptron model
# Network will have 1 input -> 1 hidden layer (8 neurons wide) -> 1 output layer.
# Modle is fit using Men Squared error, which if we take the square root gives us an error core in the units of the
# dataset

model = Sequential()
model.add(Dense(8, input_dim=1, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)

# Estimate model performance
trainscore = model.evaluate(trainX, trainY, verbose=0)

print('Train Score: %.2f MSE (%.2f RMSE)' % (trainscore, math.sqrt(trainscore)))
testscore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testscore, math.sqrt(testscore)))
