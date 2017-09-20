import numpy as np
import pandas as pd
import scipy.misc


def sigmoid(f_a):
    return 1 / (1 + np.exp(-f_a))

def relu(f_a):
    f_a[f_a <= 0] = 0
    return f_a

img_size = 784
epochs = 1000
learning_rate = 0.01
layers = [17000, 4, 1]

# read in the data and make each column an image
data = pd.read_csv("train.csv", dtype="float32").as_matrix()[:17000].transpose()

# ignore the first value in each column (the number the image represents) and make each value between 0 and 1
x = np.divide(data[1:], 256).transpose()

# take the first value in each column
y = data[:1].transpose()
# label the y values as 1 if the number is 1, otherwise 0
y[y != 1] = 0

# init weights and the bias
w = [None] * (len(layers) - 1)
b = [None] * (len(layers) - 1)
for i in range(len(w)):
    w[i] = np.random.randn(layers[i+1], layers[i])
    b[i] = np.zeros((layers[i+1], 1))

print '-- STARTING TRAINING --'
for i in range(epochs):
    # init loss
    j = 0

    # multiply inputs by weights and and bias
    z1 = w[0].dot(x) + b[0]
    # take sigmoid
    a1 = relu(z1)

    # multiply inputs by weights and and bias
    z2 = w[1].dot(a1) + b[1]
    # take sigmoid
    a2 = sigmoid(z2)

    # calculate the loss
    j += -np.sum(np.multiply(y, np.log(a2)) + np.multiply(1 - y, np.log(1 - a2))) / len(x)

    # derivative of z
    dz2 = a2 - y
    dw2 = w[1].dot(dz2)
    w[1] -= learning_rate * (np.sum(dw2, axis=0) / len(x))

    # update w and b
    #     w -= learning_rate * (np.sum(np.multiply(dz, x), axis=0) / len(x))
    #     b -= np.sum(dz) / len(x)

    # print loss
    print "Loss: " + str(j)
