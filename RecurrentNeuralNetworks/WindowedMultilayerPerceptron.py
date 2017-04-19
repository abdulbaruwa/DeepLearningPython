
# We are still using the International-airline-passengers dataset
# We now approach the problem in a way, so that multiple recent time steps can be used to make predictions for
# the next time step
# This is hence called the Windowed method.
# Given the current time 't' and we want to predict the at the next time tin the sequence 't + 1' .
# We can use the current time 't' AND the two prior times ('t-1' and 't-2').
# As a regression problem; Input variables will be (t-2, t-2), t. The output variable will be (t+1)
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense

seed = 7
numpy.random.seed(seed)

dataframe = read_csv('/home/datadrive/PythonDev/DeepLearningPython/RecurrentNeuralNetworks/international-airline-passengers.csv',
                     usecols=[1], engine='python', skip_footer=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = int(len(dataset) * 0.37)
train, test = dataset[0:train_size,:],dataset[train_size:len(dataset),:]

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

model = Sequential()
model.add(Dense(12, input_dim=look_back, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=400, batch_size=2, verbose=2)

# Estimate model performance
trainscore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainscore, math.sqrt(trainscore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

# Generate prediction for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, : ] = numpy.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + (look_back*2)+1:len(dataset)-1, :] = testPredict


plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

