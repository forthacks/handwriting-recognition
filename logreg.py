import numpy as np
import pandas as pd


def sigmoid(a):
    return 1 / (1 + np.exp(-a))

img_size = 784
epochs = 1
learning_rate = 1

# read in the data
raw_data = pd.read_csv("train.csv", dtype="float32").as_matrix()

# make each column an image
data = raw_data.transpose()

# take off the first value in each column (the number the image represents)
x = data[1:].transpose()

y = data[:1].transpose()
# label the y values as 1 if the number is 1, otherwise 0
y[y != 1] = 0

# init weights and the bias
w = np.zeros((1, img_size))
b = 0

for i in range(epochs):
    # init loss
    j = 0

    # multiply inputs by weights and and bias
    z = x.dot(w.transpose()) + b

    # take sigmoid
    a = sigmoid(z)

    # calculate the loss
    j += -np.sum(np.multiply(y, np.log(a)) + np.multiply(1 - y, np.log(1 - a)))

    # derivative of z
    dz = a - y

    # update w
    w += np.multiply()
