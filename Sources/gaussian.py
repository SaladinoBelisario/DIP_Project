import numpy as np


# Retrieves the nth of the Pascal's triangle(gaussian discrete approximation)
def pascal_vector(level):
    if level == 1:
        row = 1
    elif level == 2:
        row = [1, 1]
    else:
        row = [1, 1]
        for i in range(2, level):
            row = np.convolve([1, 1], row)
    return row


# Performs the nth derivative of the gaussian approximation from Pascal's triangle's nth level
def gaussian_derivative(level, n):
    if n == 0:
        derivative = pascal_vector(level)
    elif n > 0:
        derivative = np.convolve([1, -1], gaussian_derivative(level, n - 1))
    return derivative
