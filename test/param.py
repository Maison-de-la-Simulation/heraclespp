# Global parameter

import numpy as np

L = 1 # Domaine lenght
inter = 0.5 # Interface position

gamma = 5. / 3

z = (gamma - 1) / (2 * gamma)
g2 = (gamma + 1) / (2 * gamma)
g3 = 2 * gamma / (gamma - 1)
g4 = 2 / (gamma - 1)
g5 = 2 / (gamma + 1)
g6 = (gamma - 1) / (gamma + 1)
g7 = (gamma - 1) / 2
g8 = (gamma - 1)

## -------- Shock Tube
Ncell = 1_000 
dx = L / Ncell
x_exact = np.zeros(Ncell)
for i in range(Ncell):
    x_exact[i] = i * dx + (dx / 2)

# Initialisation
# Left
rho0l = 1
u0l = 0
P0l = 1
c0l = np.sqrt(gamma * P0l / rho0l)
var_int0l = np.array([rho0l, u0l, P0l, c0l])

# Right
rho0r = 0.125
u0r = 0
P0r = 0.1
c0r = np.sqrt(gamma * P0r / rho0r)
var_int0r = np.array([rho0r, u0r, P0r, c0r])
