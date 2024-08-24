import numpy as np
import cv2
import matplotlib.pyplot as plt


def convolution(inputFile, kernel, average=False, verbose=False):
    resultant = np.zeros_like(inputFile)

    if len(inputFile.shape) == 3:
        inputFile = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)

    fileRow, fileColumn = inputFile.shape
    kernelRow, kernelColumn = kernel.shape

    widthWithPad = (kernelColumn - 1) // 2
    heightWithPad = (kernelRow - 1) // 2

    paddingInputFile = np.pad(inputFile, ((heightWithPad, heightWithPad), (widthWithPad, widthWithPad)), mode='constant')

    for row in range(fileRow):
        for col in range(fileColumn):
            resultant[row, col] = np.sum(kernel * paddingInputFile[row:row + kernelRow, col:col + kernelColumn])

            if average:
                resultant[row, col] /= kernel.size

    return resultant.astype(np.uint8)
