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
    #print(f.keys())
    u_x = f['u'][0, 0, :, :]
    u_y = f['u'][1, 0, :, :]
    t = f['current_time'][()]
    iter = f['iter'][()]

print("Final time =", t, "s")
print("Iteration number =", iter )
    
u2d = np.sqrt(u_x**2 + u_y**2)

# ------------------------------------------

plt.figure(figsize=(15, 8))
plt.suptitle('Gresho Vortex')
plt.title('Speed')
plt.imshow(u2d, origin='lower', extent=[-1, 1, -1, 1])
plt.colorbar()
plt.plasma()
plt.xlabel('x'); plt.ylabel('y')
plt.show() 