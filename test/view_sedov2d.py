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
    #print(f.keys())
    rho = f['rho'][0, :, :]
    x = f['x'][()]
    y = f['y'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]

# Analytical result ------------------------

E0 = 1e5
rho0 = 1
beta = 0.868 #gamma = 5/3

r_choc = 1 / beta * (E0 * t**2 / rho0)**(1/5)

theta = np.linspace(0, 2*np.pi, len(x))
x_choc = r_choc * np.cos(theta)
y_choc = r_choc * np.sin(theta)

print("Final time =", t, "s")
print("Iteration number =", iter )

# ------------------------------------------

plt.figure(figsize=(10, 5))
plt.suptitle(f'Sedov blast wave 2d t = {t:1f} s')
plt.title('Density')
#plt.plot(x_choc, y_choc, label='Theorical radius')
plt.imshow(rho, origin='lower', extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
plt.colorbar()
plt.plasma()
plt.xlabel('x'); plt.ylabel('y')
plt.show()