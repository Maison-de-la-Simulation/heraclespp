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
dx_exact = L / Ncell
x_exact = np.zeros(Ncell)
for i in range(Ncell):
    x_exact[i] = i * dx_exact

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

## -------- Advection

nx = 100
dx_ad = 1. / nx
x_ad = np.zeros(nx+4)
for i in range(nx+4):
    x_ad[i] = i * dx_ad

# Sinusoide
rho0_ad_sin = np.zeros(nx+4)
for i in range(nx+4):
    rho0_ad_sin[i] = 1 * np.exp(-15 * ((1. / 2)  - x_ad[i])**2) # Sinusoïdale
    
# Crenel
rho0_ad_cre = np.zeros(nx+4)
for i in range(nx+4):
    if (x_ad[i]<= 0.3) :
        rho0_ad_cre[i] = 1
    elif (x_ad[i]>= 0.7):
        rho0_ad_cre[i] = 1
    else :
        rho0_ad_cre[i]= 2