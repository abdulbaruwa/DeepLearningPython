# Manual k-Fold cross-validation technique
# Provides a solid estimate of the perf of a model.
# spliting the training data set into k subset.
# Maths: k-subset -> subset of a set on n elements containing exactly k elements
#        The number of k-subsets on n elements is given by the binomial coefficint {n|k}
# Not often used for evaluating deep learning models because of the

# Pima Indians datase with 10-fold cross validation
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy

#fix the seed
seed = 7

numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt('/home/datadrive/PythonDev/DeepLearningPython/TrainingData/Pimaindiandiabetes', delimiter=',')

# split into input and output variables
X = dataset[:, 0:8]
Y = dataset[:, 8]

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

cvscores = []

for train, test in kfold.split(X, Y):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)

    # evaluate the model

    scores = model.evaluate(X[test], Y[test], verbose=0)
    print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print('%.2f%% (+/-%.2f%%)' % (numpy.mean(cvscores), numpy.std(cvscores)))
