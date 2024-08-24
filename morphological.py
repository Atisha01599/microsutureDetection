from erode import erosion
from dilate import dilation
import numpy as np


kernel1=np.ones((1, 16), np.uint8)
threshold1 = np.ones((3, 1), np.uint8)
threshold4 = np.ones((1, 3), np.uint8)
kernel2=np.ones((1, 10), np.uint8)

def morphOpen1(image):
    global kernel1
    eroded_image = erosion(image, kernel1)
    dilated_erode = dilation(eroded_image, kernel1)
    global threshold1
    dilated_erode = dilation(dilated_erode, threshold1)
    global threshold4
    dilated_erode = dilation(dilated_erode, threshold4)
    dilated_erode = dilation(dilated_erode, threshold1)
    dilated_erode = dilation(dilated_erode, threshold4)
    dilated_erode = dilation(dilated_erode, threshold1)
    return dilated_erode


def morphOpen2(image):
    global kernel2
    global threshold4
    global threshold1
    eroded_image = erosion(image, kernel2)
    dilated_erode = dilation(eroded_image, kernel2)
    global threshold4
    dilated_erode = dilation(dilated_erode, threshold4)
    global threshold1
    dilated_erode = dilation(dilated_erode, threshold1)
    dilated_erode = dilation(dilated_erode, threshold1)
    return dilated_erode