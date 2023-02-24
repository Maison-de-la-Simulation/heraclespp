# Test the shcok tube problem and compare 

import numpy as np
import matplotlib.pyplot as plt
import h5py
from argparse import ArgumentParser
from pathlib import Path

from param import *
from shockTube_exact import CI, ExactShockTube

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

# Condition for pressure > 0
if (g4 * (c0l + c0r)) < (u0r - u0l) :
    print('Il faut inclure le vide')

print("********************************")
print(" Shock tube problem")
print("********************************")

# Initialisation
tab_rho0, tab_u0, tab_P0 = CI(x_exact, inter, var_int0l, var_int0r)
tab_e0 = tab_P0 / tab_rho0 / g8

# Solution Exact
timeout = 0.2
rho_exact, u_exact, P_exact, e_exact = ExactShockTube(x_exact, inter, var_int0l, var_int0r, timeout)
print('Final time shock tube problem = ', timeout, 's')

# Solution solver
with h5py.File(args.filename, 'r') as f :
    print(f.keys())
    rho = f['rho'][:, 0, 0]
    u = f['u'][:, 0, 0]
    P = f['P'][:, 0, 0]
e = P / rho / (gamma - 1)

dx = L / len(rho)
x = np.zeros(len(rho))
for i in range(0, len(rho)):
    x[i] = i * dx + dx / 2

plt.figure(figsize=(10,8))
plt.suptitle('Shock tube')
plt.subplot(221)
plt.plot(x_exact, tab_rho0, '--', label='t=0')
plt.plot(x_exact,rho_exact, label='Exact')
plt.plot(x, rho, label='Solver')
plt.ylabel('Densité'); plt.xlabel('Position')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()
plt.subplot(222)
plt.plot(x_exact, tab_u0, '--', label='t=0')
plt.plot(x_exact, u_exact, label='Exact')
plt.plot(x, u, label='Solver')
plt.ylabel('Speed'); plt.xlabel('Position')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()
plt.subplot(223)
plt.plot(x_exact, tab_P0,'--', label='t=0')
plt.plot(x_exact, P_exact, label='Exact')
plt.plot(x, P, label='Solver')
plt.ylabel('pressure'); plt.xlabel('Position')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()
plt.subplot(224)
plt.plot(x_exact, tab_e0,'--', label='t=0')
plt.plot(x_exact, e_exact, label='Exact')
plt.plot(x, e, label='Solver')
plt.ylabel('Internal energy'); plt.xlabel('Position')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()
if args.output is None:
    plt.show()
else:
    plt.savefig(args.output)
