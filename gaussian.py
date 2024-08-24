import numpy as np
import math
from convol import convolution




def gaussianblur(inputFile, kernel_size, verbose=False):
    def dnorm(x, mu, sd):
        return 1 / (np.sqrt(2 * np.pi) * sd) * np.exp(-np.power((x - mu) / sd, 2) / 2)

    def gaussian_kernel(size, sigma=1, verbose=False):
        kernel1 = np.linspace(-(size // 2), size // 2, size)
        for i in range(size):
            kernel1[i] = dnorm(kernel1[i], 0, sigma)
        kernel2 = np.outer(kernel1.T, kernel1.T)

        kernel2 *= 1.0 / kernel2.max()

        return kernel2

    kernel = gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size), verbose=verbose)
    return convolution(inputFile, kernel, average=True, verbose=verbose)


