# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 12:15:30 2021

@author: mclea
"""

import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer


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


def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    """
    creata_fractal iterates over all the pixels in the image, computing the
    complex coordinates from the pixel coordinates and calls the mandel
    function at each pixel. The return valye of mandel is used to color the
    pixel.

    Parameters
    ----------
    min_x : TYPE
        DESCRIPTION.
    max_x : TYPE
        DESCRIPTION.
    min_y : TYPE
        DESCRIPTION.
    max_y : TYPE
        DESCRIPTION.
    image : TYPE
        DESCRIPTION.
    iters : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    height = image.shape[0]
    width = image.shape[0]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_x - min_y) / height

    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color


image = np.zeros((1024, 1536), dtype=np.uint8)
start = timer()
create_fractal(-2.0, 1.0, -1.0, 1.0, image, 20)
dt = timer() - start
print("Mandelbrot created in %f s", dt)
imshow(image)
show()
