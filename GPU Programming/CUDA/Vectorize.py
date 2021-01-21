# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 13:00:41 2021

@author: mclea
"""

from numba import cuda, jit, vectorize
import numpy
from pylab import imshow, show
from timeit import default_timer as timer
import matplotlib.pyplot as plt


def escape_time(p, maxtime):
    """Perform the Mandelbrot iteration until it's clear that p diverges
    or the maximum number of iterations has been reached.
    """
    z = 0j
    for i in range(maxtime):
        z = z ** 2 + p
        if abs(z) > 2:
            return i
    return maxtime

# maxiter = 200
# rlim = (-2.2, 1.5)
# ilim = (-1.5, 1.5)
# nx = 100
# ny = 75

# dx = (rlim[1] - rlim[0]) / nx
# dy = (ilim[1] - ilim[0]) / ny

# M = numpy.zeros((ny, nx), dtype=int)

# for i in range(ny):
#     for j in range(nx):
#         p = rlim[0] + j * dx + (ilim[0] + i * dy) * 1j
#         M[i, j] = escape_time(p, maxiter)

# plt.imshow(M, interpolation="nearest")

# @cuda.jit
# def my_kernel:
#     x, y, z = cuda.gird(3)

escape_time_gpu = cuda.jit(device=True)(escape_time)

@cuda.jit
def mandelbrot_gpu(M, real_min, real_max, imag_min, imag_max):
    ny, nx = M.shape
    i, j = cuda.grid(2)

    if i < ny and j < nx:
        dx = (real_max - real_min)/nx
        dy = (imag_max - imag_min)/ny
        p = real_min + dx*i + (imag_min + dy*j)*1j
        M[i, j] = escape_time_gpu(p, 2000)


M = numpy.zeros((16364, 16364), dtype=numpy.int32)
block = (32, 32)
grid = (M.shape[0] // block[0] if M.shape[0] % block[0] == 0
        else M.shape[0] // block[0] + 1,
        int(M.shape[0] // block[1] if M.shape[1] % block[1] == 0
            else M.shape[1] // block[1] + 1))

mandelbrot_gpu[grid, block](M, -2.2, 1.5, -1.5, 1.5)
plt.imshow(M)
