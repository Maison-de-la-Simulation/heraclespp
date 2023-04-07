# View Sedov blast wave 1d

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
from argparse import ArgumentParser
from pathlib import Path

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
    x = f['x'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]

gamma = 1.4
L = 1

etot = 1 / 2 * rho * u**2 + P / (gamma - 1)

dx = L / len(rho)
xc = np.zeros(len(rho))
for i in range(len(rho)):
    xc[i] = x[i] + dx / 2

# Analytical result ------------------------

rho0 = 1
E0 = 1
r, rho_exact, u_exact, P_exact = ExactSedov(rho0, E0, t, gamma)
E = 1 / 2 * rho_exact * u_exact**2 + P_exact * (gamma - 1)

print("Final time =", t, "s")
print("Iteration number =", iter )

# ------------------------------------------

plt.figure(figsize=(15,4))
plt.suptitle('Sedov blast wave 1d')
plt.subplot(131)
plt.plot(r,rho_exact, label='Exact')
plt.plot(xc, rho, label='Solver')
plt.ylabel('Density'); plt.xlabel('Position')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.xlim(0,1)
plt.grid()
plt.legend()
plt.subplot(132)
plt.plot(r, u_exact, label='Exact')
plt.plot(xc, u, label='Solver')
plt.ylabel('Velocity'); plt.xlabel('Position')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.xlim(0,1)
plt.grid()
plt.legend()
plt.subplot(133)
plt.plot(r, P_exact, label='Exact')
plt.plot(xc, P, label='Solver')
plt.ylabel('Pressure'); plt.xlabel('Position')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.xlim(0,1)
plt.grid()
plt.legend()
""" plt.subplot(224)
plt.plot(r, E, label='Exact')
plt.plot(xc, etot / 2, label='Solver')
plt.ylabel('Energy'); plt.xlabel('Position')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.xlim(0,1)
plt.grid()
plt.legend() """
plt.show()