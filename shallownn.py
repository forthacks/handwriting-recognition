import numpy as np
import pandas as pd
import scipy.misc
import sklearn.datasets


def sigmoid(f_a):
    return 1 / (1 + np.exp(-f_a))


def relu(f_a, deriv=False):
    f_a[f_a <= 0] = 0
    if deriv:
        f_a[f_a > 0] = 1
    return f_a

img_size = 784
epochs = 100
learning_rate = 0.05
layers = [img_size, 2, 1]

# data = sklearn.datasets.make_hastie_10_2()
# x = data[0].T
# y = ((data[1] + 1) / 2).T

# read in the data and make each column an image
data = pd.read_csv("train.csv", dtype="float32").as_matrix()[:17000].transpose()

# ignore the first value in each column (the number the image represents) and make each value between 0 and 1
x = np.divide(data[1:], 256)

# take the first value in each column
y = data[:1]
# label the y values as 1 if the number is 1, otherwise 0
y[y != 1] = 0

# init weights and the bias
w = [None] * (len(layers) - 1)
b = [None] * (len(layers) - 1)
for i in range(len(w)):
    w[i] = np.random.randn(layers[i+1], layers[i]) * 0.001
    b[i] = np.zeros((layers[i+1], 1))

print '-- STARTING TRAINING --'
for i in range(epochs):

    z1 = w[0].dot(x) + b[0]
    a1 = relu(z1)

    z2 = w[1].dot(a1) + b[1]
    a2 = sigmoid(z2)

    # calculate the loss
    j = -np.sum(np.dot(y, np.log(a2).T) + np.dot(1 - y, np.log(1 - a2).T)) / len(x)

    # derivatives of layer 2
    dz2 = a2 - y
    dw2 = dz2.dot(a1.T) / len(x)
    w[1] -= learning_rate * (dw2 / len(x))
    b[1] -= (learning_rate * np.sum(dz2, axis=1, keepdims=True) / len(x))

    # derivatives of layer 1
    dz1 = relu(z1, deriv=True) * w[1].T.dot(dz2)
    dw1 = dz1.dot(x.T) / len(x)
    w[0] -= learning_rate * (dw1 / len(x))
    b[0] -= (learning_rate * np.sum(dz1, axis=1, keepdims=True) / len(x))

    # print loss
    print "Gen: " + str(i) + " Loss: " + str(j)

# read in test data
test_data = pd.read_csv("train.csv", dtype="float32").as_matrix()[10000:].T

# repeat above steps for test data
test_x = np.divide(test_data[1:], 256)

test_y = test_data[:1]
test_y[test_y != 1] = 0

z1 = w[0].dot(test_x) + b[0]
a1 = relu(z1)
z2 = w[1].dot(a1) + b[1]
a2 = sigmoid(z2)

for i in range(len(a2[0])):
    print "Prediction: " + str(a2[0][i])
    print "Actual: " + str(int(test_y[0][i]))
    print

# test data img hand drawn by one of the authors!
try:

    print 'Test data image hand-drawn by one of the authors!'

    test_image = np.divide(scipy.misc.imread("Test Image.jpg", flatten=True), 256).flatten().T

    z1 = w[0].dot(test_image) + b[0]
    a1 = relu(z1)
    z2 = w[1].dot(a1) + b[1]
    a2 = sigmoid(z2)

    print "Prediction: " + str(a2[0])
    print "Actual: " + str(1)

except AttributeError:
    print 'Error. Perhaps you are missing Pillow?'
    print '$ pip install pillow'
except IOError:
    print 'Error. Perhaps you are missing libjpeg?'
    print '$ brew install libjpeg'
