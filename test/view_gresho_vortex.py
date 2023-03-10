# View Gresho Vortex problem

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

print("********************************")
print("Gresho Vortex problem")
print("********************************")

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f :
    print(f.keys())
    rho = f['rho'][0, 0, 0]
    u_x = f['u'][0, 0, :, :]
    u_y = f['u'][1, 0, :, :]
    P_x = f['P'][0, 0, :]
    P_y = f['P'][0, :, 0]
    Pr = f['P'][0, :, :]
    
dx = 1 / len(P_x)
dy = 1 / len(P_y)
x = np.zeros(len(P_x))
y = np.zeros(len(P_y))
for i in range(0, len(P_x)):
    x[i] = i * dx + dx / 2
for i in range(0, len(P_y)):
    y[i] = i * dy + dy / 2

r = np.sqrt(x**2 + y**2)
theta = np.arctan(y / x)

## 1D ---------------------------

Pr_1d = np.sqrt(P_x**2 + P_y**2)

utheta = - np.sin(theta) * u_x + np.cos(theta) * u_y
utheta_1d = np.sqrt(utheta[0]**2 + utheta[1]**2)

## 2D ---------------------------

utheta_2d = np.sqrt(u_x**2 + u_y**2)

""" plt.figure(figsize=(15, 8))
plt.suptitle('Gresho Vortex 1d')
plt.subplot(121)
plt.title('Vitesse')
plt.plot(r, utheta_1d)
plt.xlabel('r')
plt.ylabel('v$_{\theta}$')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4])
plt.grid()
plt.subplot(122)
plt.title('Pression')
plt.plot(r, Pr_1d)
plt.xlabel('r')
plt.ylabel('P(r)')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4])
plt.grid()
plt.show() """

plt.figure(figsize=(15, 8))
plt.suptitle('Gresho Vortex 2d')
plt.subplot(121)
plt.title('Vitesse')
plt.imshow(utheta_2d, origin='lower', extent=[-1, 1, -1, 1])
plt.plasma()
plt.xlabel('x'); plt.ylabel('y')
plt.subplot(122)
plt.title('Pression')
plt.imshow(Pr, origin='lower', extent=[-1, 1, -1, 1])
plt.plasma()
plt.xlabel('x'); plt.ylabel('y')
plt.show() 