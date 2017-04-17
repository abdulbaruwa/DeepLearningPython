# We have seen that a simple CNN performs poorly on this complex problem. In this section
# we look at scaling up the size and complexity of our model. Let’s design a deep version of the
# simple CNN above. We can introduce an additional round of convolutions with many more
# feature maps. We will use the same pattern of Convolutional, Dropout, Convolutional and Max
# Pooling layers.
# This pattern will be repeated 3 times with 32, 64, and 128 feature maps. The effect will be
# an increasing number of feature maps with a smaller and smaller size given the max pooling
# layers. Finally an additional and larger Dense layer will be used at the output end of the
# network in an attempt to better translate the large number feature maps to class values. We
# can summarize a new network architecture as follows:
# 1. Convolutional input layer, 32 feature maps with a size of 3 × 3 and a rectifier activation
# function.
# 2. Dropout layer at 20%.
# 3. Convolutional layer, 32 feature maps with a size of 3 × 3 and a rectifier activation function.
# 4. Max Pool layer with size 2 × 2.
# 5. Convolutional layer, 64 feature maps with a size of 3 × 3 and a rectifier activation function.
# 6. Dropout layer at 20%.
# 7. Convolutional layer, 64 feature maps with a size of 3 × 3 and a rectifier activation function.
# 8. Max Pool layer with size 2 × 2.
# 9. Convolutional layer, 128 feature maps with a size of 3 × 3 and a rectifier activation function.
# 10. Dropout layer at 20%.
# 11. Convolutional layer, 128 feature maps with a size of 3 × 3 and a rectifier activation function.
# 12. Max Pool layer with size 2 × 2.
# 13. Flatten layer.
# 14. Dropout layer at 20%.
# 15. Fully connected layer with 1,024 units and a rectifier activation function.
# 16. Dropout layer at 20%.
# 17. Fully connected layer with 512 units and a rectifier activation function.
# 18. Dropout layer at 20%.
# 19. Fully connected output layer with 10 units and a softmax activation function.

import numpy
from multi_gpu import make_parallel
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# One hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024,  activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

print(model.summary())

# Compile the model
lrate = 0.01
epochs = 200
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Fit the model (train)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)
# Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2ff%%' % (scores[1]*100))

