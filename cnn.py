import numpy as np
import pandas as pd
import math


def im2col(img, size, stepsize=1):
    a = np.lib.pad(img, ((0, (size - len(img)) % size), (0, (size - len(img[0])) % size)), "constant", constant_values=0)
    # Parameters
    m, n = a.shape
    s0, s1 = a.strides
    nrows = m-size+1
    ncols = n-size+1
    shp = size, size, nrows, ncols
    strd = s0, s1, s0, s1

    out_view = np.lib.stride_tricks.as_strided(a, shape=shp, strides=strd)
    return out_view.reshape(size*size, -1)[:, ::stepsize]


def convolve(img, filter):
    new_img = im2col(img, math.sqrt(len(filter)))
    new_img = new_img.reshape((len(new_img[0]), len(new_img)))
    new_img = np.dot(new_img, filter)
    return new_img.reshape((len(img), len(img[0])))


def convolve_layer(layer, filters, new_layer_size):
    layerNum = len(layer)
    layerSize = [len(layer[0]), len(layer[0][0])]
    outputLayer = np.zeros((new_layer_size, layerSize[0], layerSize[1]))
    for i in range(new_layer_size):
        for j in range(layerNum):
            outputLayer[j] = np.add(outputLayer[j], convolve(layer[j], filters[j][i]))
    return outputLayer


def maxPool(img, size):
    print(img)
    newImg = im2col(img, size, stepsize=size)
    newImg = np.amax(newImg, axis=0)
    print(newImg)
    return newImg.reshape((len(img) + (size - len(img)) % size, len(img[0]) + (size - len(img[0])) % size))

def pool_layer(layer, size):
    layerNum = len(layer)
    # layerSize = [len(layer[0]), len(layer[0][0])]
    outputLayer = [None] * layerNum
    for i in range(layerNum):
        outputLayer[i] = maxPool(layer[i], size)
    return outputLayer


def relu_layer(layer, leak=0.01):
    gradients = 1. * (layer > 0)
    gradients[gradients == 0] = leak
    return gradients


dataset = pd.read_csv("train.csv", dtype="float32").as_matrix()

image = np.divide(dataset[100][1:].reshape((28, 28)), 255);

def init_filters(layerSizes, size):
    filters = [None] * (len(layerSizes) - 1)
    for i in range(len(layerSizes) - 1):
        filters[i] = 0.01 * np.random.random((layerSizes[i], layerSizes[i+1], size*size)) - 0.005
    return filters


layerSizes = [1, 1, 1, 1]

filters = init_filters(layerSizes, 3)

print("Image: " + str(image))

layer1 = convolve_layer([image], filters[0], layerSizes[1])
print("Layer 1: " + str(layer1))
# layer1_pool = pool_layer(layer1, 2)
# layer1_relu = relu_layer(layer1_pool)
# layer2 = convolve_layer(layer1_relu, filters[1], layerSizes[2])
# layer2_pool = pool_layer(layer2, 2)
# layer2_relu = relu_layer(layer2_pool)
# layer3 = convolve_layer(layer2_relu, filters[2], layerSizes[3])
# print(layer3)

# test im2col
# print(im2col(np.array([[6, 5, 4], [1, 2, 3]]), 2, stepsize=2))
