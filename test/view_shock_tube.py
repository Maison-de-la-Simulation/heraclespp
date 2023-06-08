# Test the shock tube problem and compare

import numpy as np
import matplotlib.pyplot as plt
import h5py
from argparse import ArgumentParser
from pathlib import Path

from exact_shockTube import CI, ExactShockTube

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
print("Shock tube problem")
print("********************************")

with h5py.File(args.filename, 'r') as f :
    print(f.keys())
    rho = f['rho'][0, 0, :]
    u = f['u'][0, 0, 0, :]
    P = f['P'][0, 0, :]
    x = f['x'][()]
    fx = f['fx'][0, 0, 0, :]
    t = f['current_time'][()]
    iter = f['iter'][()]
    gamma = f['gamma'][()]

L = np.max(x) - np.min(x)

e = P / rho / (gamma - 1)

dx = np.zeros(len(rho))
for i in range(len(rho)):
    dx[i] = x[i+1] - x[i]

xc = np.zeros(len(rho))
for i in range(len(rho)):
    xc[i] = x[i] + dx[i] / 2

print("Final time =", t, "s")
print("Iteration number =", iter )

print("Max fx =", np.max(fx), fx.shape)

print("dx = ", dx)

# Initialisation ------------------------
inter = 0.5 # Interface position
# Left
rho0l = 1
u0l = 0
P0l = 1
c0l = np.sqrt(gamma * P0l / rho0l)
var0l = np.array([rho0l, u0l, P0l, c0l])

# Right
rho0r = 0.125
u0r = 0
P0r = 0.1
c0r = np.sqrt(gamma * P0r / rho0r)
var0r = np.array([rho0r, u0r, P0r, c0r])

Ncell = 1_000 
dx_exact = L / Ncell
x_exact = np.zeros(Ncell+1)
for i in range(len(x_exact)):
    x_exact[i] = i * dx_exact

rho0, u0, P0 = CI(x_exact, inter, var0l, var0r)
e0 = P0 / rho0 / (gamma - 1)

# Analytical result ------------------------

rho_exact, u_exact, P_exact, e_exact = ExactShockTube(x_exact, inter, var0l, var0r, t, gamma)

# ------------------------------------------

plt.figure(figsize=(10,8))
plt.suptitle(f'Shock tube t = {t:1f} s')
plt.subplot(221)
plt.plot(x_exact, rho0, '--', label='t=0')
plt.plot(x_exact,rho_exact, label='Exact')
plt.plot(xc, rho, 'x-', label =f't = {t:1f}')
plt.plot(xc, fx, label='scalar')
plt.ylabel('Density'); plt.xlabel('Position')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()
plt.subplot(222)
plt.plot(x_exact, u0, '--', label='t=0')
plt.plot(x_exact, u_exact, label='Exact')
plt.plot(xc, u, 'x-', label =f't = {t:1f}')
plt.ylabel('Velocity'); plt.xlabel('Position')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()
plt.subplot(223)
plt.plot(x_exact, P0,'--', label='t=0')
plt.plot(x_exact, P_exact, label='Exact')
plt.plot(xc, P, 'x-', label =f't = {t:1f}')
plt.ylabel('Pressure'); plt.xlabel('Position')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()
plt.subplot(224)
plt.plot(x_exact, e0,'--', label='t=0')
plt.plot(x_exact, e_exact, label='Exact')
plt.plot(xc, e, 'x-', label =f't = {t:1f}')
plt.ylabel('Internal energy'); plt.xlabel('Position')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()

plt.figure()
plt.title('Passive scalar')
plt.plot(xc, fx)
plt.grid()

if args.output is None:
    plt.show()
else:
    plt.savefig(args.output)

