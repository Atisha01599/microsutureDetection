import cv2
import numpy as np


def addWeighted(inputFile1, alpha, inputFile2, beta, gamma, dtype=np.uint8):
    dtype = dtype or inputFile1.dtype
    
    if inputFile1.shape != inputFile2.shape:
        raise ValueError("give correct image")

    output = inputFile1 * alpha
    output += inputFile2 * beta
    output += gamma

    output = np.clip(output, 0, 255)
    output = output.astype(dtype)

    return output