#Jared Staman
#CS 425: Assignment 3

import skimage
from skimage import io
from matplotlib import pyplot as plt

image = io.imread('baboon.jpg')
r = image[:,:,0]
g = image[:,:,1]
b = image[:,:,2]

fig, (ax1,ax2,ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(15,4))

ax1.imshow(r, cmap="gray")

ax2.imshow(g, cmap="gray")

ax3.imshow(b, cmap="gray")

ax4.imshow(image)
plt.show()