from keras.datasets import cifar10
from matplotlib import pyplot
import scipy.misc as ta

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# create a grid of 3X3 images
for i in range(0, 9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(ta.toimage(X_train[i]))
    pyplot.imshow(image_array, cmap='Greys', interpolation='None')
pyplot.show()