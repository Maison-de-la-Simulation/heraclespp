# Test the shock tube problem and compare

from argparse import ArgumentParser
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from exact_shock_tube import CI, ExactShockTube

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
print("         Shock tube")
print("********************************")

with h5py.File(args.filename, 'r') as f:
    #print(f.keys())
    rho = f['rho'][0, 0, :]
    u = f['ux'][0, 0, :]
    P = f['P'][0, 0, :]
    x = f['x'][()]
    fx = f['fx'][0, 0, 0, :]
    T = f['T'][0, 0, :]
    t = f['current_time'][()]
    iter = f['iter'][()]
    gamma = f['gamma'][()]

print(f"Final time = {t:.1f} s")
print(f"Iteration number = {iter}")

print("Max fx =", np.max(fx))

xmin = x[2]
xmax = x[len(x)-3]
L = xmax - xmin

e = P / rho / (gamma - 1)

dx = np.zeros(len(rho))
for i in range(2, len(rho)+2):
    dx[i-2] = x[i+1] - x[i]

xc = np.zeros(len(rho))
for i in range(2, len(rho)+2):
    xc[i-2] = x[i] + dx[i-2] / 2

print("dx =", dx)

# Analytical result ------------------------

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
x_exact = np.zeros(Ncell)

for i in range(len(x_exact)):
    x_exact[i] = i * dx_exact + dx_exact / 2

rho0, u0, P0 = CI(x_exact, inter, var0l, var0r)
e0 = P0 / rho0 / (gamma - 1)

rho_exact, u_exact, P_exact, e_exact = ExactShockTube(x_exact, inter, var0l, var0r, t, gamma)

# ------------------------------------------

plt.figure(figsize=(10,8))
plt.suptitle(f'Shock tube t = {t:.1f} s')
plt.subplot(221)
plt.plot(x_exact, rho0, '--', label='t=0')
plt.plot(x_exact,rho_exact, label='Exact')
plt.plot(xc, rho, label = 'Solver')#f't = {t:1f}')
plt.plot(xc, fx, label='Scalar')
plt.xlabel('Position')
plt.ylabel('Density ($kg.m^{-3}$)')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()

plt.subplot(222)
plt.plot(x_exact, u0, '--', label='t=0')
plt.plot(x_exact, u_exact, label='Exact')
plt.plot(xc, u, 'x-', label = 'Solver')#f't = {t:1f}')
plt.xlabel('Position')
plt.ylabel('Velocity ($m.s^{-1}$)')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()

plt.subplot(223)
plt.plot(x_exact, P0,'--', label='t=0')
plt.plot(x_exact, P_exact, label='Exact')
plt.plot(xc, P, 'x-', label = 'Solver')#f't = {t:1f}')
plt.xlabel('Position')
plt.ylabel('Pressure ($kg.m^{-1}.s^{-2}$)')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()

plt.subplot(224)
plt.plot(x_exact, e0,'--', label='t=0')
plt.plot(x_exact, e_exact, label='Exact')
plt.plot(xc, e, 'x-', label = 'Solver')#f't = {t:1f}')
plt.xlabel('Position')
plt.ylabel('Internal energy ($m^{2}.s^{-2}$)')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()

plt.figure()
plt.title('Passive scalar')
plt.plot(xc, fx)
plt.grid()

plt.figure()
plt.title('Temperature')
plt.plot(xc, T)
plt.grid()

if args.output is None:
    plt.show()
else:
    plt.savefig(args.output)
