# ** NOTE Got errors in sclearn when running this. **

# We will use the Iris classification Dataset
# the 4 inputs are numeric
# Sepal length, Sepal width, petal length, petal width, class (Lengths in centimeters)

# Remember a Multiclassification problem just means there are more than two classes to be predicted.
# in this case, there are three flower species.

import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


seed = 7
numpy.random.seed(seed)

dataframe = read_csv('/home/datadrive/PythonDev/DeepLearningPython/MulticlassClassification/irisdata', header=None)
dataset = dataframe.values
# take first 4 items for each row convert to float
X = dataset[:, 0:4].astype(float)

# take last item - which is the classifier
Y = dataset[:, 4]

# lets encode the output variables
# We reshape from vector that contains values for each class value to; a matrix with a
# boolean for each class value. And whether or not a given instance has that class
# value or not. 'One hot encoding'
# for out classification of:
#   Iris-setosa
#   Iris-versicolor
#   Iris-virginica
#
# One-hot encoded to
#   Iris-setosa,     Iris-versicolor,     Iris-virginica
#        1,               0,                     0
#        0,               1,                     0
#        0,               0,                     1

# Encode string to integers using scikit-learn class LabelEncoder
# then convert vector of integers to a One-hot encoding using keras 'to_categorical()' function
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert the integers to dummy variables - one-hot encoded
dummy_y = np_utils.to_categorical(encoded_Y)
# print(dummy_y)

# We define a fully connected network with the topology
# 4 inputs nodes -> [4 hidden nodes] -> 3 output nodes
# inputs nodes matches match the iris properties (Sepal length, Sepal width, petal length, petal width) in our dataset

# define baseline model
def baseline_model():
    print('Start build for baseline model')
    model = Sequential()
    model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(3, kernel_initializer='normal', activation='sigmoid'))

    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


print('Creating classifier')
# create keras classifier for use in scikit learn.
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

print('Shuffle and create 10-folds')
# Evaluate the model with k-fold cross-validation
# setting the number of folds to 10, also shuffling the data before partitioning our fold sets
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

print('Evaluating ')
# #  lets use 10-fold cross validation to evaluate our model on our dataset X and dummy_y.
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# print('Accuracy: %.2f%% (%.2f%%)' % (results.mean()*100, results.std()*100))





