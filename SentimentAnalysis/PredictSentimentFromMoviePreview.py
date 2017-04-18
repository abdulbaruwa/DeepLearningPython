# Looking how to use word embedding in Keras for natural language problems.

import numpy
from keras.datasets import imdb
from matplotlib import pyplot

# load
(X_train, y_train), (X_test, y_test) = imdb.load_data(path='/home/datadrive/PythonDev/DeepLearningPython/SentimentAnalysis/imdb_full.pkl')
X = numpy.concatenate((X_train, X_test), axis=0)
y = numpy.concatenate((y_train, y_test), axis=0)

# summarize size
print('Training data: ')
print(X.shape)
print(y.shape)

# Summarize number of classes
print("Classes: ")
print(numpy.unique(y))
# Summarize number of words
print("Number of words: ")
print(len(numpy.unique(numpy.hstack(X))))
# Summarize review length
print("Review length: ")
result = [len(x) for x in X]
print(result)
print("Mean %.2f words (%f)" % (numpy.mean(result), numpy.std(result)))

pyplot.subplot(121)
pyplot.boxplot(result)
pyplot.show()