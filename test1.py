import numpy as np
import cv2 as cv

#sphere radius...
r = 50

'meshgrid the z(x,y)=sqrt(r^2-(x^2+y^2))'
X = np.linspace(-1.5 * r, 1.5 * r, 30 * r)
Y = np.copy(X)
[x,y] = np.meshgrid(X, Y)

#print([x,y])

#surface albedo
albedo = 0.5

#illumination direction...
I = [0.2, 0, 0.98]

#surface partial derivates at each point
resd = r**2 - (x**2 + y**2)
mask = (resd >= 0)
resd = resd * mask

index = np.where(resd == 0)
resd[index[0],index[1]] = 1

p = -x / np.sqrt(resd)
q = -y / np.sqrt(resd)

#now calculate the brightness as each point
R = (albedo * (I[0] * p - I[1] * q + I[2])) / np.sqrt(1 + p**2 + q**2)

R = R * mask
index = np.where(R < 0)
R[index[0], index[1]] = 0

R = R / np.max(R) * 255

cv.imwrite("test1.jpg",R)
