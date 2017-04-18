# WORD Embeddings
# Technique where words are encoded as real-value vectors in dimensional space.
# The similarity between meanings of words is translates to closeness in the vector space.
# Discrete words are mapped to vectors of continuous numbers,which is useful for when working
# with natural language problems in neural networks - as numbers are required as inputs.

# In keras we can convert positive integer representations of words into a word embedding by an Embedding layer.
# The layer takes arguments
#   * Mapping
#   * Maximum number of expected words (Vocabulary size e.g the largest integer value seen as input)
#   * Dimensionality of each word vector - Called the output dimension

# In the IMDB dtaset, we assume we are interested in the first 5,000 most used words in the dataset.
# Our vocabulary size will be 5,000.
# We can choose to use a 32-dimensional vector to represent each word
# We can cap the maximum review length at 500 words. (Truncate those more and pad those less with 0.

# About the data.
# The words have been replaced by integers that indicate the absolute popularity of the
# word in the dataset. The sentences in each review are therefore comprised of a sequence of
# integers.

# Multilayer Perceptron model
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

seed = 7
numpy.random.seed(seed)

# load the dataset but only keep the top 5000 words and zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# bound reviews at 500 words
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# create the model
model = Sequential()
model.add(Embedding(top_words, 128, input_length=max_words))
model.add(Flatten())
model.add(Dense(1500, activation='relu'))
# model.add(Dense(500, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit the model - The model overfits quickly - we will use very few training epochs.
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128, verbose=0)

# Final evaluation
scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f%%' % (scores[1]*100))

