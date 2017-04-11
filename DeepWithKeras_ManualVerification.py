from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy


seed = 7
# fix random seed for reproducibility
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt('/home/datadrive/PythonDev/DeepLearningPython/TrainingData/Pimaindiandiabetes', delimiter=',')

# split into input and output variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

# We intend to use Manual verification so will use sklearn to split training data
# split into 67% for training and 33% for testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile model
# For simplicity, we are evaluating the performance of the network on the same data set. Ideally we should
# have a 'training' and 'test' data sets.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# let us train (Fit) the model. This will run a number of epochs.
model.fit(X_train, y_train, epochs=150, batch_size=10, validation_data=(X_test, y_test))

# evaluate the model
scores = model.evaluate(X, Y)
print('\n%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))


