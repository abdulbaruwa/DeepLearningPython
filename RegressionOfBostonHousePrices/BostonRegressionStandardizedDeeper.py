# Attempt to improving performance by evaluating a deeper Network topology
import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import model_from_json
import os

dataframe = read_csv('/home/datadrive/PythonDev/DeepLearningPython/RegressionOfBostonHousePrices/boston', delim_whitespace=True, header=None)
dataset = dataframe.values

X = dataset[:, 0:13]
Y = dataset[:, 13]

def larger_model():
    model = Sequential()
    model.add(Dense(30, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    # compile
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

modeltosave = larger_model()
model_json = modeltosave.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

modeltosave.save_weights("model.h5")
print('Model saved to disk')


seed = 7
numpy.random.seed(seed)

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=100, batch_size=5, verbose=5)))

pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)

print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))

