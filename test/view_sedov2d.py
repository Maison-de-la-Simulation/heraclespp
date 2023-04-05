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
    
print("Final time =", t, "s")
print("Iteration number =", iter )

theta = np.linspace(0, 2*np.pi, len(x))

# Analytical result ------------------------

E0 = 10**5
rho0 = 1
alpha = 0.049

r_choc = (E0 / alpha *rho0)**(1/5) * t**(2/5) 
x_choc = r_choc * np.cos(theta)
y_choc = r_choc * np.sin(theta)
r = np.sqrt(x_choc**2 + y_choc**2)

# ------------------------------------------

plt.figure(figsize=(15, 5))
plt.suptitle('Sedov blast wave 2d')
plt.subplot(121)
plt.title('Density')
#plt.plot(x_choc, y_choc, 'o')
plt.imshow(rho, origin='lower', extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
plt.colorbar()
plt.plasma()
plt.xlabel('x'); plt.ylabel('y')
plt.show() 