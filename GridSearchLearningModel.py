# Trying Grid search (sklearn lib) to evaluate different configuration for a neural network.
# looking for and reporting the combination that provides the most acceptable estimated performance.


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy

# create a function to create a model. Required by KerasClassifier.
# The function is defined with two args 'optimizer' and 'init' -> must be defaulted
# This args will enable us evaluate the effect of using different optimization algorithms and
# weight initialization schemes for the network.

def create_model(optimizer='rmsprop', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer=init, activation='relu'))
    model.add(Dense(8, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))

    # compile
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# define a random seed
seed = 7
numpy.random.seed(seed)

print('Loading dataset')
# load pima indians dataset
dataset = numpy.loadtxt('/home/datadrive/PythonDev/DeepLearningPython/TrainingData/Pimaindiandiabetes', delimiter=',')

# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

print('Creating model classifier')
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)

# grid search: epochs, batch size and optimizer
optimizers = ['rmsprop', 'adam']
inits = ['glorot_uniform', 'normal', 'uniform']
epochs = [50, 100, 150]
batches = [5, 10, 20]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=inits)
grid = GridSearchCV(estimator=model, param_grid=param_grid)

print('Train the model with the grid')
grid_result = grid.fit(X, Y)

print('Grab results')
# summarize result

print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print('%f (%f) with: %r' % (mean, stdev, param))

