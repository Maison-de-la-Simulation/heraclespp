# View Sedov blast wave 2d

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

print("********************************")
print("Sedov blast wave 2d")
print("********************************")

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f :
    print(f.keys())
    rho = f['rho'][0, :, :]
    rho_x = f['rho'][0, 0, :]
    rho_y = f['rho'][0, :, 0]
    P = f['P'][0, :, :]

x = np.linspace(-5, 5, len(rho_x))
y = np.linspace(-5, 5, len(rho_y))
theta = np.linspace(0, 2*np.pi, len(rho_x))

t = 0.025
E0 = 10**5
rho0 = 1
alpha = 0.049

r_choc = (E0 / alpha *rho0)**(1/5) * t**(2/5) 
x_choc = r_choc * np.cos(theta)
y_choc = r_choc * np.sin(theta)
r = np.sqrt(x_choc**2 + y_choc**2)

plt.figure(figsize=(15, 8))
plt.suptitle('Sedov blast wave 2d')
plt.subplot(121)
plt.title('Density')
#plt.plot(x_choc, y_choc, 'o')
plt.imshow(rho, origin='lower', extent=[-5, 5, -5, 5])
plt.colorbar()
plt.plasma()
plt.xlabel('x'); plt.ylabel('y')
plt.show() 