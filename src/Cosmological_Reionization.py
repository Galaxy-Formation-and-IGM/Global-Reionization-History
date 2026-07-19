### Basic package import

import numpy as np

### src imports

import Cosmology as cosmo
import Constants_Units as cu

### SFRD models from literature

def Madau(z):
    return 0.01 * (1+z)**2.6 / ( 1+((1+z)/3.2)**6.2 )

def Harikane(z):
    return 1 / ( 61.7 * (1+z)**(-3.13) + 10**(0.22*(1+z)) + 2.4 * 10**(0.5*(1+z)-3) )

def New_SFR(z): #Khaire
    return 10 ** (-2) * (2.01 + 8.48 * z) / (1 + (z / 2.5) ** 3.09)

### Reionization problem constants

n_gamma = 4800 #Number of ionizing photons produced per baryon (proton), dimensionless
n_gamma2 = 7780
f_esc = 0.1 #Fraction of photons escaping the star forming halo
f_esc2 = 0.2

### Self-written RK4

def RK4(f, x_0, y_0, h, x_n):
    steps = int(np.abs(np.abs(x_n - x_0) / h))
    x_arr = np.zeros(steps + 1)
    y_arr = np.zeros_like(x_arr)
    
    x_arr[0] = x_0
    y_arr[0] = y_0
    counter = 0

    for i in range(1, steps + 1):
        counter += 1
        k_1 = f(x_arr[i - 1], y_arr[i - 1])
        k_2 = f(x_arr[i - 1] + h / 2, y_arr[i - 1] + h * k_1 / 2)
        k_3 = f(x_arr[i - 1] + h / 2, y_arr[i - 1] + h * k_2 / 2)
        k_4 = f(x_arr[i - 1] + h, y_arr[i - 1] + h * k_3)

        y_arr[i] = y_arr[i - 1] + h * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
        x_arr[i] = x_arr[i - 1] + h

        if y_arr[i] >= 1:
            for j in range(counter, steps + 1):
                y_arr[j] = 1
                x_arr[j] = x_arr[j - 1] + h
            break
    return x_arr, y_arr

### Reionization functions

alpha_B = 2.59 * 10 ** (-13) #Case B recombination coefficient at T=3*10^4 K in cm^3 * s^-1

alpha_B_unit = alpha_B * (cu.cmToMpc) ** 3 / (cu.sToYr) #Coefficient in Mpc^3 * yr^-1

def n_H(z): #Mean proper number denisty of hydrogen atoms in Mpc^-3
    return cosmo.X_H * cosmo.Om_b * cosmo.rho_c * (1 + z) ** 3 / (cu.m_p * cu.kgToSM)

def dt_dz(z): #In yr
    return -1 / ((1 + z) * cosmo.H(z))

def dN_dt_Mad(z):
    return Madau(z) * (1 + z) ** 3 * n_gamma2 * f_esc / (cu.m_p * cu.kgToSM)

def Clump(z):
    #if z >= 6:
     #   return 1 + 9 * (7 / (1 + z)) ** 2
    #else:
     #   return 10
    #return 3
    return 2.9 * ((1 + z) / 6) ** (-1.1)

def RHS_Mad(z, f_HII):
    return dN_dt_Mad(z) * dt_dz(z) / n_H(z) - alpha_B_unit * n_H(z) * f_HII * Clump(z) * dt_dz(z)

### Solving reionization problem 

Mad_data = RK4(RHS_Mad, 20, 0, -0.1, 0.01)