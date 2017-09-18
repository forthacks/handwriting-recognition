import numpy as np
import pandas as pd
import scipy.misc


def sigmoid(f_a):
    return 1 / (1 + np.exp(-f_a))


img_size = 784
epochs = 1000
learning_rate = 0.01

# read in the data and make each column an image
data = pd.read_csv("train.csv", dtype="float32").as_matrix()[:17000].transpose()

# ignore the first value in each column (the number the image represents) and make each value between 0 and 1
x = np.divide(data[1:].transpose(), 256)

# take the first value in each column
y = data[:1].transpose()
# label the y values as 1 if the number is 1, otherwise 0
y[y != 1] = 0

# init weights and the bias
w = np.zeros((1, img_size))
b = 0

print '-- STARTING TRAINING --'

for i in range(epochs):
    # init loss
    j = 0

    # multiply inputs by weights and and bias
    z = x.dot(w.transpose()) + b

    # take sigmoid
    a = sigmoid(z)
    # calculate the loss
    j += -np.sum(np.multiply(y, np.log(a)) + np.multiply(1 - y, np.log(1 - a))) / len(x)

    # derivative of z
    dz = a - y

    # update w and b
    w -= learning_rate * (np.sum(np.multiply(dz, x), axis=0) / len(x))
    b -= np.sum(dz) / len(x)

    # print loss
    print "Loss: " + str(j)

print
print '-- STARTING TESTING --'
print

# read in test data and divide by 256 to make values between 0 and 1
test_data = np.divide(pd.read_csv("test.csv", dtype="float32").as_matrix()[10000:], 256).transpose()

# repeat above steps for test data
test_x = test_data[1:].transpose()

test_y = test_data[:1].transpose()
y[y != 1] = 0

z = x.dot(w.transpose()) + b
a = sigmoid(z)

for i in range(len(a)):
    print "Prediction: " + str(a[i][0])
    print "Actual: " + str(int(y[i][0]))
    print

# test data img hand drawn by one of the authors!
try:

    print 'Test data image hand-drawn by one of the authors!'

    test_image = np.divide(scipy.misc.imread("Test Image.jpg", flatten=True), 256).flatten()
    z = test_image.dot(w.transpose()) + b
    a = sigmoid(z)

    print "Prediction: " + str(a[0])
    print "Actual: " + str(1)

except AttributeError:
    print 'Error. Perhaps you are missing Pillow?'
    print '$ pip install pillow'
except IOError:
    print 'Error. Perhaps you are missing libjpeg?'
    print '$ brew install libjpeg'
