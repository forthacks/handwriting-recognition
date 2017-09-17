import numpy as np
import pandas as pd
from scipy import misc

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

img_size = 784
epochs = 10
learning_rate = 0.01

# read in the data
train_data = pd.read_csv("train.csv", dtype="float32").as_matrix()[:17000]

# make each column an image
data = train_data.transpose()

# ignore the first value in each column (the number the image represents) and make each value between 0 and 1
x = np.divide(data[1:].transpose(), 256)

# take the first value in each column
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
    j += -np.sum(np.multiply(y, np.log(a)) + np.multiply(1 - y, np.log(1 - a))) / len(x)

    # derivative of z
    dz = a - y

    # update w and b
    w -= learning_rate * (np.sum(np.multiply(dz, x), axis=0) / len(x))
    b -= np.sum(dz) / len(x);

    # print loss
    print("Loss: " + str(j))


# read in test data and divide by 256 to make values between 0 and 1
test_data = np.divide(pd.read_csv("train.csv", dtype="float32").as_matrix()[10000:], 256).transpose()

# repeat above steps for test data
test_x = test_data[1:].transpose()

test_y = test_data[:1].transpose()
y[y != 1] = 0

z = x.dot(w.transpose()) + b
a = sigmoid(z)
for i in range(len(a)):
    print("Prediction: " + str(a[i]))
    print("Actual: " + str(y[i]))


# test data img hand drawn by on of the authors!
test_image = np.divide(misc.imread("Test Image.jpg", flatten=True), 256).flatten()
z = test_image.dot(w.transpose()) + b
a = sigmoid(z)
print("Prediction: " + str(a))
print("Actual: " + str(1))
