# View Sedov blast wave 2d

import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py

print("********************************")
print("      Sedov blast wave 2d")
print("********************************")

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f:
    #print(f.keys())
    rho = f['rho'][0, :, :]
    x = f['x'][()]
    y = f['y'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]
    gamma = f['gamma'][()]

print(f"Final time = {t:.1e} s")
print(f"Gamma = {gamma}")

xmin = x[2]
xmax = x[len(x)-3]
ymin = y[2]
ymax = y[len(y)-3]

# Analytical result ---------------------------------------------------------- #

E1 = 1e5
rho0 = 1
beta = 1.15 # gamma = 5/3

r_choc = beta * (E1 * t**2 / rho0)**(1/5)

theta = np.linspace(0, 2*np.pi, len(x))
x_choc = r_choc * np.cos(theta)
y_choc = r_choc * np.sin(theta)

# ---------------------------------------------------------------------------- #

plt.figure(figsize=(10, 5))
plt.title(f'Sedov blast wave 2d t = {t:1f} s')
plt.plot(x_choc, y_choc, label='Theoretical radius')
plt.imshow(rho, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='inferno')
plt.colorbar(label=r"$\rho$")
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

plt.show()