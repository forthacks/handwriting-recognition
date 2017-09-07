import numpy as np
import pandas as pd


def im2col(img, size, stepsize=1):
    A = np.lib.pad(img, ((0,((size-len(img)) % size)), (0, (size-len(img[0])) % size)), "constant", constant_values=0)
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides
    nrows = m-size+1
    ncols = n-size+1
    shp = size,size,nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(size*size,-1)[:,::stepsize]


def convolve(img, filter):
    newImg = im2col(img, len(filter))
    print(len(newImg))
    print(len(newImg[0]))
    print(newImg)
    print(filter)
    newImg = np.multiply(newImg, filter)
    return newImg.reshape((len(img),len(img[0])))


def convolveLayer(layer, filters, newLayerNum):
    layerNum = len(layer)
    layerSize = [len(layer[0]),len(layer[0][0])]
    outputLayer = np.zeros((newLayerNum, layerSize[0],layerSize[1]))
    for j in range(newLayerNum):
        for i in range(layerNum):
            outputLayer[i]=np.add(outputLayer[i],convolve(layer[j], filters[i][j]))
    return outputLayer


def maxPool(img, size):
    newImg = im2col(img, size, stepsize=size)
    newImg = np.amax(newImg, axis=1)
    return newImg.reshape((len(img),len(img[0])))


def poolLayer(layer, size):
    layerNum = len(layer)
    layerSize = [len(layer[0]),len(layer[0][0])]
    outputLayer = [None]*layerNum
    for i in range(layerNum):
        outputLayer[i] = maxPool(layer[i], size)
    return outputLayer


def reluLayer(layer, leak=0.01):
    gradients = 1. * (layer > 0)
    gradients[gradients == 0] = leak
    return gradients


dataset = np.divide(pd.read_csv("train.csv").as_matrix(), 256)*2 - 1

img = dataset[0][1:].reshape((28,28))
print(img)


def initFilters(layerSizes, size):
    filters = [None]*(len(layerSizes)-1)
    for i in range(len(layerSizes)-1):
        filters[i] = 0.01*np.random.random((layerSizes[i],layerSizes[i+1], size*size))-0.005
    return filters


layerSizes=[1, 10, 10, 4]
filters=initFilters(layerSizes, 3)
layer1 = convolveLayer([img], filters[0], layerSizes[1])
layer1_pool = poolLayer(layer1)
layer1_relu = reluLayer(layer1_pool)
layer2 = convolveLayer(layer1_relu, filters[1], layerSizes[2])
layer2_pool = poolLayer(layer2)
layer2_relu = reluLayer(layer2_pool)
layer3 = convolveLayer(layer2_relu, filters[2], layerSizes[3])
print(layer3)

#test im2col
print(im2col(np.array([[6, 5, 4], [1, 2, 3]]), 2, stepsize=2))
