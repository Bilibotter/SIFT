import math
import numpy as np
from PIL import Image


# -23
def readImg(path='lena.jpg'):
    pixels = Image.open(path)
    return np.array(pixels)


def template(radius=3, sigma=1.0):
    weights_matrix = np.empty((radius*2+1, 3))

    for i in range(2*radius+1):
        span = normal(i-radius, sigma)
        weights_matrix[i] = [span]*3
    return 3 * weights_matrix / np.sum(weights_matrix)


def normal(i, j, sigma=11.0):
    return math.exp(-(i*i+j*j)/2/sigma/sigma)/2/math.pi/sigma/sigma


radius = 3
img = readImg()
split_gauss_matrix = np.empty(img.shape)
temp = template(radius=radius, sigma=1.5) / 2
half_temp = temp[:radius+1] * 4
reverse_temp = temp[radius:] * 4
height, width = img.shape[:2]
precent = lambda x: '\r{:.2%}'.format(x / (height - 2 * radius + 2))
for i in range(radius, height-radius):
    print(precent(i), end='')
    for j in range(radius, width-radius):
        vertical = img[i-radius:i+radius+1, j, :]
        horizontal = img[i, j-radius:j+radius+1, :]
        v = np.multiply(vertical, temp).T
        h = np.multiply(horizontal, temp).T
        split_gauss_matrix[i][j] = np.array([np.sum(v[0]+h[0]),
                                             np.sum(v[1]+h[1]),
                                             np.sum(v[2]+h[2])])
print('\r')
res = Image.fromarray(np.uint8(split_gauss_matrix))
res.show()
