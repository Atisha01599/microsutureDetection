import numpy as np
import math



def findAngle(x1, y1, x2, y2):
    if x2 == x1:
      return 90
    else:
        angletemp = math.atan2(y2 - y1, x2 - x1)
        resultantdegree = math.degrees(angletemp)
        return resultantdegree
        
    
def findCentroid(x1, y1, x2, y2):
    y = (y1 + y2) // 2
    x = (x1 + x2) // 2
    
    return x, y

def findBinary(src, thresh_val):

    image = np.zeros_like(src, dtype=np.uint8)
    image[src <= thresh_val] = 255

    return image


def findAngelVariance(x, y):

    if len(x) < 2:
        return 0.0

    sum1= sum((val - y) ** 2 for val in x)

    length = len(x) - 1
    Angelvariance = sum1/ length

    return Angelvariance

def findAngelMean(x):

    if len(x) == 0:
        print("Error")
        return 0.0

    return sum(x) / len(x)
