# View Sedov blast wave 1d

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
from argparse import ArgumentParser
from pathlib import Path

from init_sedov1d import gamma, rho0_sed, E0_sed_per, t_sed, gamma, L
from exact_sedov import ExactSedov

parser = ArgumentParser(description="Plot shock tube.")
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
print("Sedov blast wave 1d")
print("********************************")

filename = sys.argv[1]

with h5py.File(args.filename, 'r') as f :
    print(f.keys())
    rho = f['rho'][0, 0, :]
    u = f['u'][0, 0, 0, :]
    P = f['P'][0, 0, :]
e = P / rho / (gamma - 1)

dx = L / len(rho)
x = np.zeros(len(rho))
for i in range(0, len(rho)):
    x[i] = i * dx + dx / 2

r, rho_exact, u_exact, P_exact = ExactSedov(rho0_sed, E0_sed_per, t_sed)

print(len(r))

plt.figure(figsize=(15,4))
plt.suptitle('Sedov blast wave 1d')
plt.subplot(131)
plt.plot(r,rho_exact, label='Exact')
plt.plot(x, rho, label='Solver')
plt.ylabel('Density'); plt.xlabel('Position')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.xlim(0,1)
plt.grid()
plt.legend()
plt.subplot(132)
plt.plot(r, u_exact, label='Exact')
plt.plot(x, u, label='Solver')
plt.ylabel('Velocity'); plt.xlabel('Position')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.xlim(0,1)
plt.grid()
plt.legend()
plt.subplot(133)
plt.plot(r, P_exact, label='Exact')
plt.plot(x, P, label='Solver')
plt.ylabel('Pressure'); plt.xlabel('Position')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.xlim(0,1)
plt.grid()
plt.legend()
plt.show()