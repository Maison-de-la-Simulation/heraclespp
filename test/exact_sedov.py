#           SEDOV BLAST-WAVE
#           Solution exacte
#
# Réécriture de la routine IDL : sedovana.pro

import numpy as np

def SolutionSedov(n, gamma):
    n1 = 1_000
    n2 = 1_000

    vmax = 4 / (n + 2) / (gamma + 1)
    vmin = 2 / (n + 2) / gamma

    v = np.zeros(n1)
    for i in range(n1):
        v[i] = vmin + 10**(- 10 * (1 - (i + 1) / n1)) * (vmax - vmin)

    a2 = (1 - gamma) / (2 * (gamma - 1) + n)
    a1 = (n + 2) * gamma / (2 + n * (gamma - 1)) * (2 * n *(2 - gamma) / gamma /(n + 2)**2 - a2)
    a3 = n / (2 * (gamma - 1) + n)
    a4 = a1 * (n + 2) / (2 - gamma)
    a5 = 2 / (gamma - 2)
    a6 = gamma / (2 * (gamma - 1) + n)
    a7 = a1 * (2 + n * (gamma - 1)) / (n * (2 - gamma))

    r1 = ((n + 2) * (gamma + 1) / 4 * v)**(- 2 / (2 + n)) * \
        ((gamma + 1 ) / (gamma - 1) * ((n + 2) * gamma / 2 * v - 1))**(- a2)* \
        ((n + 2) * (gamma + 1) / ((n + 2) * (gamma + 1) - 2 * (2 + n * (gamma - 1))) * \
        (1 - (2 + n * (gamma - 1)) / 2 * v))**(-a1)

    u1 = (n + 2) * (gamma + 1 ) / 4 * v * r1

    d1 = ((gamma + 1) / (gamma - 1) * ((n + 2) * gamma / 2 * v - 1))**(a3)* \
        ((gamma + 1) / (gamma - 1) * (1 -(n + 2) / 2 * v))**(a5)* \
        ((n + 2) * (gamma + 1) / ((n + 2) * (gamma + 1) - 2 * (2 + n * (gamma - 1))) * \
        (1 - (2 + n * (gamma - 1)) / 2 * v))**(a4)

    p1 = ((n + 2) * (gamma + 1) / 4 * v)**(2 * n / (2 + n)) * \
        ((gamma + 1) / (gamma - 1) * (1 - (n + 2) / 2 * v))**(a5 + 1)* \
        ((n + 2) * (gamma + 1) / ((n + 2) * (gamma + 1) - 2 * (2 + n * (gamma - 1))) *\
        (1 - (2 + n * (gamma - 1)) / 2 * v))**(a4 - 2 * a1)

    r2 = np.zeros(n2)
    for i in range(n2):
        r2[i] = r1[0] * (i + 0.5) / n2

    u2 = u1[0] * r2 / r1[0]
    d2 = d1[0] * (r2 / r1[0])**(n / (gamma - 1))
    p2 = p1[0] * (r2 / r2)

    r = np.zeros(len(r1) + len(r2) + 2)
    r[0:len(r2)] = r2
    r[len(r2):len(r1) + len(r2)] = r1
    r[-2] = np.max(r1)
    r[-1] = np.max(r1) + 1_000

    d = np.zeros(len(r))
    d[0:len(r2)] = d2
    d[len(r2):len(r1) + len(r2)] = d1
    d[-2] = 1 / ((gamma + 1) / (gamma - 1))
    d[-1] = 1 / ((gamma + 1) / (gamma - 1))

    u = np.zeros(len(r))
    u[0:len(r2)] = u2
    u[len(r2):len(r1) + len(r2)] = u1
    u[-2] = 0
    u[-1] = 0

    p = np.zeros(len(r))
    p[0:len(r2)] = p2
    p[len(r2):len(r1) + len(r2)] = p1
    p[-2] = 0
    p[-1] = 0

    dnew = d * (gamma + 1) / (gamma - 1)
    unew = u * 4 / (n + 2) / (gamma + 1)
    pnew = p * 8 / (n + 2)**2 / (gamma + 1)

    nn = len(r)
    vol = np.zeros(nn)
    for i in range(1, nn-1):
        vol[i] = r[i]**n - r[i-1]**n
    vol[0] = r[0]**n
    const = 1

    if(n==1):
        const = 2
    elif(n==2):
        const = np.pi
    elif(n==3):
        const = 4 * np.pi / 3

    u = unew
    d = dnew
    p = pnew
    vol = vol *const
    int1 = (d * u* u/ 2) * vol
    int2 = p / (gamma - 1) * vol
    sum1 = np.sum(int1)
    sum2 = np.sum(int2)
    sum = sum1 + sum2
    print('chi0 =', sum**(-1 / (2 + n)))
    chi0 = sum**(- 1 / (2 + n))
    r = r * chi0
    u = u * chi0
    p = p * chi0**2
    etot=0
    rr = np.zeros(len(r)+1)
    for i in range(1,len(rr)-1):
        rr[i] = (r[i] + r[i-1]) / 2
    rr[-1] = 2 * (r[-1] -r[-2])
    
    for i in range(len(r)):
        #etot = etot + 4 / 3 * np.pi* (rr[i+1]**3 - rr[i]**3) * (p[i] / (gamma - 1) + 1 / 2 * d[i] * u[i]**2)
         etot = etot + 2.*(rr[i+1] - rr[i]) * (p[i] / (gamma - 1) + 1 / 2 * d[i] * u[i]**2)
    print(etot)
    return r, d, u, p

def ExactSedov(rho_0, E_per, t, gamma):
    n = 1 # Cartesian
    r, rho, u, P = SolutionSedov(n, gamma)
    r_f = r * (E_per / rho_0)**(1 / (n + 2)) * t**(2 / (n + 2))
    rho_f = rho * rho_0
    u_f = u * (E_per / rho_0)**(1 / (n + 2)) * t**(- n / (n + 2))
    P_f = P * (E_per / rho_0)**(2 / (n + 2)) * t**(- 2 * n / (n + 2)) * rho_0
    return r_f, rho_f, u_f, P_f