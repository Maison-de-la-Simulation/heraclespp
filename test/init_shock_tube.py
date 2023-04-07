# Global parameter

import numpy as np

L = 1 # Domaine lenght
inter = 0.5 # Interface position
gamma = 1.4

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