### Basic packages import

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

### src imports

import Cosmology as cosmo

### Press-Schechter Formalism

z_range = np.linspace(0.1,20, 100)

f_b = cosmo.Om_b/cosmo.Om_m     # fraction of baryonic matter
H0 = cosmo.H(0)         # in units of 1/yr

def W(k,R):
    kR = k*R
    return 3*(np.sin(kR) - kR*np.cos(kR))/(kR**3)

def P(k,z,cutoff=np.inf):
    if k<=cutoff:    
        return Pi(k)*(T(k)*D(z))**2
    else:
        return 0

# Primordial power spectrum (un-normalised)
def Pi(k):
    return k ** cosmo.n

# Transfer Function
def T(k):
    Gamma = cosmo.Om_m * cosmo.h * np.exp( -cosmo.Om_b * (1 + np.sqrt(2 * cosmo.h) / cosmo.Om_m) )
    q = k / (Gamma * cosmo.h)
    value = 1 + (3.89 * q) + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4
    return np.log(1 + 2.34 * q) / (2.34 * q * value ** (1 / 4))

def D(z):
    Dz = (cosmo.Om_m + 0.4545 * cosmo.Om_lambda) / (cosmo.Om_m * (1 + z) ** 3 + 0.4545 * cosmo.Om_lambda)
    return Dz ** (1 / 3)

# Variance of smoothed density field (wrt radius, R)
def sigma_R(R, z = 0, cutoff = 1000):    
    # Integrand for the variance calculation
    def integral(k):
        return P(k, z, cutoff) * k ** 2 * W(k, R) ** 2 / (2 * np.pi ** 2)
    
    sigma2, _ = quad(integral, 0, np.inf, limit=500, epsabs=1e-5, epsrel=1e-5)
    return np.sqrt(sigma2)


# Variance of smoothed density field (wrt mass of region, M)
def sigma_M(M,z=0,cutoff=1000):
    # Compute R corresponding to the mass M
    R = (3 * M / (4 * np.pi * cosmo.rho_0))**(1/3)
    
    return sigma_R(R,z,cutoff)

# Evaluate and print the normalization factor
R8 = 8 / cosmo.h      # units of Mpc

A = cosmo.sigma8_obs / sigma_R(R8,0)

M_range = np.logspace(6, 16, 500)  # Mass range in solar masses

y = [A*sigma_M(M,0,np.inf) for M in M_range]
y_1 = [A*sigma_M(M,0,1) for M in M_range]
y_10 = [A*sigma_M(M,0,10) for M in M_range]

def poly_fit(x_arr,y_arr,degree,plot=0):
    # Perform polynomial fit
    coefficients = np.polyfit(x_arr, y_arr, degree)

    # Generate the fitted polynomial function
    polynomial = np.poly1d(coefficients)

    # Plot the original data and the fitted polynomial
    if (plot):
        plt.figure(figsize=(12, 7))
        plt.plot(x_arr, polynomial(x_arr), 'r', lw=1.5, label=f'Polynomial Fit (Degree {degree})')
        plt.plot(x_arr, y_arr, '--',color='k', label='Original Data')
        plt.xscale('log')
        plt.yscale('log')  
        plt.xlabel('$\log M$')
        plt.ylabel('$\sigma_o(M)$')
        plt.legend()
        plt.title('Polynomial Fit')
        plt.show()
    
    return coefficients

degree = 9

logM = np.log(M_range)

sigma_range = [A*sigma_M(M) for M in M_range]

cof = poly_fit(logM,sigma_range,degree=9)
d_cof = np.polyder(cof)

fit_deriv = np.poly1d(d_cof)
    
def PS_MassFunc(M, z):
    # Generate the fitted polynomial function
    polynomial = np.poly1d(cof)
    sigma0 = polynomial(np.log(M))
    d_sigma = abs(fit_deriv(np.log(M)))
    factor = np.sqrt(2 / np.pi) * cosmo.delta_c/(sigma0**2 * D(z)) * abs(fit_deriv(np.log(M)))
    exponent = (-cosmo.delta_c**2 / (2 * sigma0**2 * D(z)**2))
    return factor * np.exp(exponent)