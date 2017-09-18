import math

import numpy as np
import pandas as pd


def im2col(img, size, step_size=1):
    a = np.lib.pad(img, ((0, (size - len(img)) % size), (0, (size - len(img[0])) % size)), "constant",
                   constant_values=0)
    # Parameters
    m, n = a.shape
    s0, s1 = a.strides
    rows = m - size + 1
    cols = n - size + 1
    shp = size, size, rows, cols
    strides = s0, s1, s0, s1

    out_view = np.lib.stride_tricks.as_strided(a, shape=shp, strides=strides)
    return out_view.reshape(size * size, -1)[:, ::step_size]


def convolve(img, f_filter):
    new_img = im2col(img, math.sqrt(len(f_filter)))
    new_img = new_img.reshape((len(new_img[0]), len(new_img)))
    new_img = np.dot(new_img, f_filter)
    return new_img.reshape((len(img), len(img[0])))


def convolve_layer(layer, f_filters, new_layer_size):
    layer_num = len(layer)
    layer_size = [len(layer[0]), len(layer[0][0])]
    output_layer = np.zeros((new_layer_size, layer_size[0], layer_size[1]))
    for i in range(new_layer_size):
        for j in range(layer_num):
            output_layer[j] = np.add(output_layer[j], convolve(layer[j], f_filters[j][i]))
    return output_layer


def max_pool(img, size):
    new_img = im2col(img, size, step_size=size)
    new_img = np.amax(new_img, axis=0)
    print(new_img)
    return new_img.reshape((len(img) + (size - len(img)) % size, len(img[0]) + (size - len(img[0])) % size))


def pool_layer(layer, size):
    layer_num = len(layer)
    # layer_size = [len(layer[0]), len(layer[0][0])]
    output_layer = [None] * layer_num
    for i in range(layer_num):
        output_layer[i] = max_pool(layer[i], size)
    return output_layer


def relu_layer(layer, leak=0.01):
    gradients = 1. * (layer > 0)
    gradients[gradients == 0] = leak
    return gradients


def init_filters(f_layer_sizes, size):
    f_filters = []
    for i in range(len(f_layer_sizes) - 1):
        f_filters[i].append(0.01 * np.random.random((f_layer_sizes[i], f_layer_sizes[i + 1], size * size)) - 0.005)
    return f_filters


data = pd.read_csv("train.csv", dtype="float32").as_matrix()

image = np.divide(data[100][1:].reshape((28, 28)), 255)

layer_sizes = [1, 1, 1, 1]

filters = init_filters(layer_sizes, 3)

print "Image: " + str(image)

layer1 = convolve_layer([image], filters[0], layer_sizes[1])
print "Layer 1: " + str(layer1)
layer1_pool = pool_layer(layer1, 2)
print "Pool Layer 1: " + str(layer1_pool)


# layer1_relu = relu_layer(layer1_pool)
# layer2 = convolve_layer(layer1_relu, filters[1], layerSizes[2])
# layer2_pool = pool_layer(layer2, 2)
# layer2_relu = relu_layer(layer2_pool)
# layer3 = convolve_layer(layer2_relu, filters[2], layerSizes[3])
# print(layer3)

# test im2col
# print(im2col(np.array([[6, 5, 4], [1, 2, 3]]), 2, step_size=2))
