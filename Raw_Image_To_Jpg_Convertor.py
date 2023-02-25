import numpy as np
import math

## Image Processing libraries
import skimage
from skimage import exposure

import scipy.misc as misc

import rawpy
import imageio

## Visual and plotting libraries
import matplotlib.pyplot as plt

## Reading a RAW file:
rawImg = rawpy.imread(r'F:\Python Programs - ML\Dataset\Unzipped\Raw Files\Studio Session6-1460.cr2')
rgbImg = rawImg.postprocess()
rgbImg = rawImg.raw_image_visible

print(type(rgbImg))

def basic_showImg(img, size=4):
    '''Shows an image in a numpy.array type. Syntax:
        basic_showImg(img, size=4), where
            img = image numpy.array;
            size = the size to show the image. Its value is 4 by default.
    '''
    plt.figure(figsize=(size,size))
    plt.imshow(img)
    plt.show()
    
basic_showImg(rgbImg,8)    