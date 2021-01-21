# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 12:15:30 2021

@author: mclea
"""
import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer
from numba import cuda, jit
from numba import *


@jit
def mandel(x, y, max_iters):
    """
    Given the real and imaginary parts of a complex number, determine if it is
    a candidate for membership in the Mandelbrot set given a fixed number of
    itterations.

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    max_iters : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag+z.imag) >= 4:
            return i
    return max_iters


@cuda.jit
def mandel_kernel(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x)/width
    pixel_size_y = (max_y - min_y)/height

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    for x in range(startX, width, gridX):
        real = min_x + x*pixel_size_x
        for y in range(startY, height, gridY):
            imag = min_y + y * pixel_size_y
            image[y, x] = mandel_gpu(real, imag, iters)


mandel_gpu = cuda.jit(device=True)(mandel)

gimage = np.zeros((1024*20, 1536*20), dtype=np.uint8)
blockdim = (32, 8)
griddim = (32, 16)

start = timer()
d_image = cuda.to_device(gimage)
mandel_kernel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, d_image, 200)
d_image.to_host()
dt = timer() - start

start = timer()
d_image = cuda.to_device(gimage)
mandel_kernel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, d_image, 200)
d_image.to_host()
dt = timer() - start

print("Mandelbrot created on GPU in %f s", dt)
imshow(gimage)
show()