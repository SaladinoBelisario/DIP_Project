from gaussian import *


def gaussian_filter(kernel_size):
    f = gaussian_derivative(kernel_size, 0)
    f = f / sum(f)
    f.shape = (1, kernel_size)
    return f


def sobel_filter():
    f = gaussian_derivative(2, 1)
    f.shape = (1, 3)
    return f


def laplacian_filter(kernel_size):
    f = gaussian_derivative(kernel_size - 2, 2)
    f.shape = (1, kernel_size)
    return f


def gaussian_blur(kernel_size):
    f = gaussian_filter(kernel_size)
    f = f * np.transpose(f)
    return f
