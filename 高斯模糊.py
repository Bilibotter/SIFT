import time
import math
import numpy as np
from PIL import Image


def readImg(path='success0.jpg'):
    pixels = Image.open(path)
    return np.array(pixels)


def filter(pixels, radius=1, sigma=1, showProgress=True):
    blur_matrix = np.empty(pixels.shape)
    width, height, unuse = pixels.shape
    weight_matrix = template(radius, sigma)
    if showProgress:
        precent = lambda x: '\r{:.2%}'.format(x / (width - 2 * radius + 2))

    for i in range(radius, width - radius):
        if showProgress:
            print(precent(i), end='')
        for j in range(radius, height - radius):
            radius_matrix = pixels[i - radius:i + radius + 1, j - radius:j + radius + 1]
            pixel = np.multiply(weight_matrix, radius_matrix).transpose((2, 0, 1))
            blur_matrix[i, j] = [pixel[0].sum(), pixel[1].sum(), pixel[2].sum()]
    print('\n')
    return np.uint8(blur_matrix)


def template(radius=3, sigma=1):
    weights_matrix = np.empty((radius*2+1, radius*2+1, 3))

    for i in range(2*radius+1):
        for j in range(2*radius+1):
            span = normal(i-radius, j-radius, sigma)
            weights_matrix[i][j] = [span]*3

    return 3 * weights_matrix / np.sum(weights_matrix)


def normal(i, j, sigma=11.0):
    return math.exp(-(i*i+j*j)/2/sigma/sigma)/2/math.pi/sigma/sigma


def saveImg(blur_matrix, path='test.jpg'):
    img = Image.fromarray(np.uint8(blur_matrix))
    img.save(path)


def showImg(blur_matrix):
    img = Image.fromarray(np.uint8(blur_matrix))
    img.show()


start = time.time()

pixels = readImg(path='lena.jpg')

blur_matrix = filter(pixels, radius=2, sigma=8, showProgress=True)

saveImg(blur_matrix, path='test.jpg')

showImg(blur_matrix)
