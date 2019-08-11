import numpy as np
import cv2 as cv
import math
from scipy import signal
import numpy as np

def gradient(E):
    filterx = np.array(([-1, 1], [-1, 1]))
    filtery = np.array(([-1, -1],[1, 1]))

    Ex = signal.convolve2d(E, filterx, mode='same') / 2
    Ey = signal.convolve2d(E, filtery, mode='same') / 2

    return [Ex, Ey]

def estimate_albedo_illumination(E):
    #compute the average of the E
    mu1 = np.mean(E)

    #compute the average of the square of the brightness
    mu2 = np.mean(np.mean(E**2))

    #calculatethe gradient
    [Ex, Ey] = gradient(E)
    eps = 0.000001

    #normalize the gradient
    Exy = np.sqrt(Ex**2 + Ey**2)
    nEx = Ex / (Exy + eps)
    nEy = Ey / (Exy + eps)

    avgEx = np.mean(nEx)
    avgEy = np.mean(nEy)
    
    gamma = (6 * (np.pi**2) * mu2) - (48 * mu1**2)
    gamma = np.sqrt(gamma if gamma >= 0 else 0)
    albedo = gamma / np.pi

    #estamating the slant
    T = 4 * mu1 / (gamma + eps)
    T = -1 if T < -1 else T
    T = 1 if T > 1 else T
    slant = math.acos(T)

    #estamating the tilt
    tilt = math.atan(avgEy / avgEx)

    if tilt < 0:
        tit = tilt + np.pi

    #the illumination direction
    I = [np.cos(tilt) * np.sin(slant), np.sin(tilt) * np.sin(slant), np.cos(slant)]
    return [albedo, I, slant, tilt]
