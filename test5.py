import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util import estimate_albedo_illumination
import cv2 as cv

E = cv.imread('road_border.jpg')
E = cv.cvtColor(E, cv.COLOR_BGR2GRAY)
[M, N] = np.shape(E)
M = M
N = N
E = cv.resize(E, (N, M))
[albedo, I, slant, tilt] = estimate_albedo_illumination(E)


p = np.zeros((M, N))
q = np.zeros((M, N))

z = np.zeros((M, N))

z_x = np.zeros((M, N))
z_y = np.zeros((M, N))

[x,y] = np.meshgrid(np.linspace(1, N, N), np.linspace(1, M, M))

max_iter = 50 

ix = np.cos(tilt) * np.tan(slant)
iy = np.sin(tilt) * np.tan(slant)
eps = 0.00001
for k in range(1, max_iter):

    R = (np.cos(slant) + p * np.cos(tilt) * np.sin(slant) + q * np.sin(tilt) * np.sin(slant)) / np.sqrt(1 + p**2 + q**2)
    index = np.where(R < 0)
    R[index[0], index[1]] = 1

    f = E - R

    df_dz = (p + q) * (ix * p + iy * q + 1) / (np.sqrt((1 + p**2 + q**2)**3) * (np.sqrt(1 + ix**2 + iy**2))) - (ix + iy) / (np.sqrt(1 + p**2 + q**2) * np.sqrt(1 + ix**2 + iy**2))
    
    z = z - f / (df_dz + eps)

    z_x[1:M,:] = z[0:M-1,:]
    z_y[:,1:N] = z[:, 0:N-1]

    p = z - z_x
    q = z - z_y

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, z)
plt.savefig('test5_road_border.jpg')
plt.show()
