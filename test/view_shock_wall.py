# View shock on wall 1d

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser(description="Plot shock wall.")
parser.add_argument('filename',
                    type=Path,
                    help='Input filename')
parser.add_argument('-o', '--output',
                    required=False,
                    default=None,
                    type=Path,
                    help='Output file to generate')
args = parser.parse_args()

print("********************************")
print("Shock wall 1d")
print("********************************")

filename = sys.argv[1]

with h5py.File(args.filename, 'r') as f :
    print(f.keys())
    rho = f['rho'][0, 0, :]
    u = f['u'][0, 0, 0, :]
    P = f['P'][0, 0, :]
    x = f['x'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]
 
L = np.max(x) - np.min(x)

dx = np.zeros(len(rho))
for i in range(len(rho)):
    dx[i] = x[i+1] - x[i]

xc = np.zeros(len(rho))
for i in range(len(rho)):
    xc[i] = x[i] + dx[i] / 2

print("Final time =", t, "s")
print("Iteration number =", iter)

# ------------------------------------------

plt.figure(figsize=(15,4))
plt.suptitle(f'Shock wall 1d t = {t:1f} s')
plt.subplot(131)
plt.plot(xc, rho, 'x',label='Solver')
plt.ylabel('Density'); plt.xlabel('Position')
plt.grid()
plt.legend()
plt.subplot(132)
plt.plot(xc, u, 'x', label='Solver')
plt.ylabel('Velocity'); plt.xlabel('Position')
plt.grid()
plt.legend()
plt.subplot(133)
plt.plot(xc, P, 'x', label='Solver')
plt.ylabel('Pressure'); plt.xlabel('Position')
plt.grid()
plt.legend()
""" plt.subplot(224)
plt.plot(xc, etot / 2, label='Solver')
plt.ylabel('Energy'); plt.xlabel('Position')
plt.xlim(0,1)
plt.grid()
plt.legend() """
plt.show()