# Improving performance
# Neutral networks are suitable for having consistent input values both in Scale and Distribution.
# We can use Standardization as effective data preparation scheme for tabular data when building network models.
# How? Data is rescaled such that the mean value for each attribute is 0 and the standard deviation is 1.
# Doing this preserves Gaussian and Gaussian-like (normal distribution) distributions whilst normalizing the central tendencies for each
# attribute.
# It is good practice to train the standardization procedure on the training data within the pass of a cross-validation
# run and to use the trained standardization instance to prepare the unseen test fold
# This makes standardization a step in the model preparation in the cross validation procedure.
# We will achieve this using Scikit-learn's Pipeline class. The pipeline is a wrapper that executes one or more models
# within a pass of the cross-validation procedure.

import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


seed = 7
numpy.random.seed(seed)

dataframe = read_csv("/home/datadrive/PythonDev/DeepLearningPython/BinaryClassificationSonarReturns/sonar.csv", header=None)
dataset = dataframe.values


# split the data into 60 input variables
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]

# output values are string, convert them into 0, 1
encoder = LabelEncoder()
encoder.fit(Y)
encoded_y = encoder.transform(Y)
print(Y)
print(encoded_y)

# create keras model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_y, cv=kfold)

# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


