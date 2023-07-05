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
    u = f['ux'][0, 0, :]
    P = f['P'][0, 0, :]
    x = f['x'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]
    gamma = f['gamma'][()]
 
L = np.max(x) - np.min(x)

dx = np.zeros(len(rho))
for i in range(len(rho)):
    dx[i] = x[i+1] - x[i]

xc = np.zeros(len(rho))
for i in range(len(rho)):
    xc[i] = x[i] + dx[i] / 2

e = P / rho / (gamma - 1)

print("Final time =", t, "s")
print("Iteration number =", iter)

# ------------------------------------------
plt.figure(figsize=(14,9))
plt.suptitle(f'Shock wall 1d t = {t:1f} s')
plt.plot(xc, rho, 'x', label='Solver')
plt.ylabel('Density ($kg.m^{-3}$)'); plt.xlabel('x')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()

plt.figure(figsize=(14,9))
plt.suptitle(f'Shock wall 1d t = {t:1f} s')
plt.subplot(221)
plt.plot(xc, rho,label='Solver')
plt.ylabel('Density ($kg.m^{-3}$)'); plt.xlabel('x')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()
plt.subplot(222)
plt.plot(xc, u, label='Solver')
plt.ylabel('Velocity ($m.s^{-1}$)'); plt.xlabel('x')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()
plt.subplot(223)
plt.plot(xc, P, label='Solver')
plt.ylabel('Pressure ($kg.m^{-1}.s^{-2}$)'); plt.xlabel('x')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()
plt.subplot(224)
plt.plot(xc, e, label='Solver')
plt.ylabel('Internal energy ($m^{2}.s^{-2}$)'); plt.xlabel('x')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()
plt.show()
