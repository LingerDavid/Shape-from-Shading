import numpy as np
import cv2 as cv
import math
from util import gradient

#read the image
E = cv.imread('test1.jpg')
E = cv.cvtColor(E, cv.COLOR_BGR2GRAY)

#normalize the image
E = E / np.max(E)

#compute the average of the E
mu1 = np.mean(E)

#compute the average of the square of the brightness
mu2 = np.mean(np.mean(E**2))

#calculatethe gradient
[Ex, Ey] = gradient(E)
eps = 0.001

#normalize the gradient
Exy = np.sqrt(Ex**2 + Ey**2)
nEx = Ex / (Exy + eps)
nEy = Ey / (Exy + eps)

avgEx = np.mean(nEx)
avgEy = np.mean(nEy)

gamma = np.sqrt((6 * np.pi**2 * mu2) - (48 * mu1**2))
albedo = gamma / np.pi

#estamating the slant
slant = math.acos(4 * mu1 / gamma)

#estamating the tilt
tilt = math.atan(avgEy / avgEx)

if tilt < 0:
    tit = tilt + np.pi

#the illumination direction
I = [np.cos(tilt) * np.sin(slant), np.sin(tilt) * np.cos(slant), np.cos(slant)]

print(I)
