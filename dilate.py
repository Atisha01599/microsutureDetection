import numpy as np

def dilation(inputFile, kernel):
    kheight = kernel.shape[0]
    kwidth = kernel.shape[1]
    height = inputFile.shape[0]
    width = inputFile.shape[1]
    result = np.zeros((height, width), dtype=np.uint8)
    print(result)
    padded_inputFile = np.pad(inputFile, ((kheight // 2, kheight // 2), (kwidth // 2, kwidth // 2)), mode='constant')
    i = 0
    while i < height:
        j = 0
        while j < width:
            result[i, j] = np.max(padded_inputFile[i:i + kheight, j:j + kwidth] * kernel)
            j += 1
        i += 1

    return result