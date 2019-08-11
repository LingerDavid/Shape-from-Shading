import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util import estimate_albedo_illumination
import cv2 as cv

plt.style.use('ggplot')

E = cv.imread('cup.jpeg') 
E = cv.cvtColor(E, cv.COLOR_BGR2GRAY)

E = E / np.max(E)

[albedo, I, slant, tilt] = estimate_albedo_illumination(E)

Fe = np.fft.fft(E)

[M, N] = np.shape(E)
[x, y] = np.meshgrid(np.linspace(1, N, N), np.linspace(1, M, M))
wx = (2 * np.pi * x) / M
wy = (2 * np.pi * y) / N

eps = 0.0001
Fz = Fe / (-1j * wx * np.cos(tilt) * np.sin(slant) - 1j * wy * np.sin(tilt) * np.sin(slant) + eps)

z = np.abs(np.fft.ifft(Fz))

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, z)
plt.savefig('test4_cup.jpg')
plt.show()
