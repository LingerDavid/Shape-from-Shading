import numpy as np
import cv2 as cv
from scipy import signal
from util import estimate_albedo_illumination
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util import gradient
plt.style.use('ggplot')

#read the imageE = cv.imread('cup.jpeg')
E = cv.imread('test1.jpg')
E = cv.cvtColor(E, cv.COLOR_BGR2GRAY)

#normalizing the image
E = cv.resize(E, (500, 500))
E = E / np.max(E)

[albedo, I, _, _] = estimate_albedo_illumination(E)

[M, N] = np.shape(E)

#surface normals
p = np.zeros((M,N))
q = np.zeros((M,N))

p_ = np.zeros((M, N))
q_ = np.zeros((M, N))

z = np.zeros((M, N))

index = np.where(E > 0.75)
z[index[0],index[1]] = -100 * E[index[0], index[1]]
[p, q] = gradient(z)

#the estimated reflectance map
R = np.zeros((M, N))
#the controling parameter
my_lambda = 1000

#maximum number of iterations
max_iter = 50

#filter
w = 0.25 * np.array(([0, 1, 0], [1, 0, 1], [0, 1, 0]))

#wx and wy
[x, y] = np.meshgrid(np.linspace(1, N, N), np.linspace(1, M, M))
#todo
wx = (2 * np.pi * x) / N
wy = (2 * np.pi * y) / M

for k in range(1 , max_iter):
    #the second order derivates
    p_ = signal.convolve2d(p, w, mode='same')
    q_ = signal.convolve2d(q, w, mode='same')

    #compute the reflectance map
    R = (albedo * (-I[0] * p - I[1] * q + I[2])) / np.sqrt(1 + p**2 + q**2)

    pq = 1 + p**2 + q**2

    dR_dp = (-albedo * I[0] / (pq**0.5)) + (-I[0] * albedo * p - I[1] * albedo * q + I[2] * albedo) * (-1 * p * (pq**-1.5))

    dR_dq = (-albedo * I[1] / (pq**0.5)) + (-I[0] * albedo * p - I[1] * albedo * q + I[2] * albedo) * (-1 * q * (pq**-1.5))

    p = p_ + (1 / (4 * my_lambda)) * (E - R) * dR_dp
    q = q_ + (1 / (4 * my_lambda)) * (E - R) * dR_dq

    Cp = np.fft.fft(p)
    Cq = np.fft.fft(q)

    C = -1j * (wx * Cp + wy * Cq) / (wx**2 + wy**2)

    z = np.abs(np.fft.ifft(C))

    p = np.fft.ifft(1j * wx * C)
    q = np.fft.ifft(1j * wy * C)

    if k % 10 == 0:
        np.save('{0}.npy'.format(k), z)

fig = plt.figure()
ax = Axes3D(fig)
index = np.where(z > 1)
z[index[0], index[1]] = 0
ax.plot_surface(x, y, z)
plt.savefig('test3.jpg')
plt.show()
