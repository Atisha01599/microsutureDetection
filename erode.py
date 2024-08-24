import numpy as np

def erosion(inputFile, kernel):
    height = inputFile.shape[0]
    width = inputFile.shape[1]
    kernelh = kernel.shape[0]
    kernelw = kernel.shape[1]
    result = np.zeros((height, width), dtype=np.uint8)
    print(kernelh,kernelw)
    paddingAdded = np.pad(inputFile, ((kernelh // 2, kernelh // 2), (kernelw // 2, kernelw // 2)), mode='constant')

    i = 0
    while i < height:
        j = 0
        while j < width:
            result[i, j] = np.min(paddingAdded[i:i + kernelh, j:j + kernelw] * kernel)
            j += 1
        i += 1

    return result