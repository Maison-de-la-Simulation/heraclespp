# View shock on wall 1d

from argparse import ArgumentParser
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

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
print("        Shock wall")
print("********************************")

filename = sys.argv[1]

with h5py.File(args.filename, 'r') as f:
    #print(f.keys())
    rho = f['rho'][0, 0, :]
    u = f['ux'][0, 0, :]
    P = f['P'][0, 0, :]
    x = f['x'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]
    gamma = f['gamma'][()]

print("Final time =", t, "s")
print("Iteration number =", iter)

xmin = x[2]
xmax = x[len(rho)+2]
L = xmax - xmin

dx = np.zeros(len(rho))
for i in range(2, len(rho)+2):
    dx[i-2] = x[i+1] - x[i]

xc = np.zeros(len(rho))
for i in range(2, len(rho)+2):
    xc[i-2] = x[i] + dx[i-2] / 2

e = P / rho / (gamma - 1)

# ------------------------------------------
plt.figure(figsize=(14,9))
plt.suptitle(f'Shock wall 1d t = {t:.1f} s')
plt.plot(xc, rho, 'x', label='Solver')
plt.xlabel('x')
plt.ylabel('Density ($kg.m^{-3}$)')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()

plt.figure(figsize=(14,9))
plt.suptitle(f'Shock wall 1d t = {t:1f} s')
plt.subplot(221)
plt.plot(xc, rho,label='Solver')
plt.xlabel('x')
plt.ylabel('Density ($kg.m^{-3}$)')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()

plt.subplot(222)
plt.plot(xc, u, label='Solver')
plt.xlabel('x')
plt.ylabel('Velocity ($m.s^{-1}$)')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()

plt.subplot(223)
plt.plot(xc, P, label='Solver')
plt.xlabel('x')
plt.ylabel('Pressure ($kg.m^{-1}.s^{-2}$)')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()

plt.subplot(224)
plt.plot(xc, e, label='Solver')
plt.xlabel('x')
plt.ylabel('Internal energy ($m^{2}.s^{-2}$)')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()

plt.savefig("shock_wall.pdf")

plt.show()
