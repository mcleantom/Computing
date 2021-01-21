# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:13:48 2021

@author: mclea
"""

from numba import njit
import random
import time

def monte_carlo_pi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x**2 + y**2) < 1.0:
            acc += 1
    return 4.0 * acc / nsamples


start = time.perf_counter()
pi = monte_carlo_pi(1000000)
end = time.perf_counter()
print(end-start)

monte_carlo_pi_jit = njit()(monte_carlo_pi)
