# Develop network model with Keras for regression problem.
# Regression: Where data is labelled with 'real' value (say floating point) rather than a label. As opposed to what we've
# used so far 'Classification'. Where a label is a assigned a class for example spam/non-spam, fraudulent/non-fraudulent.
# Examples Time Series data like stock prices over time. The decision being modelled is what value to predict for new
# unpredicted data.


# Boston house price dataset describes properties of houseds in Boston. We are concerned with modelling the prices of
# houses in thousands of dollars.
#
# 1. CRIM: per capita crime rate by town.
# 2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
# 3. INDUS: proportion of non-retail business acres per town.
# 4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# 5. NOX: nitric oxides concentration (parts per 10 million).
# 6. RM: average number of rooms per dwelling.
# 7. AGE: proportion of owner-occupied units built prior to 1940.
# 8. DIS: weighted distances to five Boston employment centers.
# 9. RAD: index of accessibility to radial highways.
# 10. TAX: full-value property-tax rate per $ 10,000.
# 11. PTRATIO: pupil-teacher ratio by town.
# 12. B: 1000(Bk − 0.63) 2 where Bk is the proportion of blacks by town.
# 13. LSTAT: % lower status of the population.
# 14. MEDV: Median value of owner-occupied homes in $ 1000s.

import numpy
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load the data set - The dataset though is not in csv format.
dataframe = read_csv('/home/datadrive/PythonDev/DeepLearningPython/RegressionOfBostonHousePrices/boston',
                     delim_whitespace=True, header=None)
dataset = dataframe.values

# split the dataset into input (X) and output (Y) variables for easier modelling with keras & scikit learn

X = dataset[:, 0:13]
print(X[0])
Y = dataset[:, 13]
print(Y[0])

# define the baseline model
def baseline_model():
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# #Evaluate model as a regression model
# #estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)

# Evaluate with standardized dataset:
# Use sklearn Pipeline to perform standardization during model evaluation process, within each fold of
# the cross-validation.
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=70, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)

kfold = KFold(n_splits=10, random_state=seed)
# #results = cross_val_score(estimator, X, Y, cv=kfold)
results = cross_val_score(pipeline, X, Y, cv=kfold)

print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))



