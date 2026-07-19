### Basic packages import

import numpy as np
from numba import jit

### Basic Cosmology used throughout the project

h = 0.67 

Om_m = 0.315            # matter density
Om_b = 0.022/(h*h)      # baryonic matter density
Om_lambda = 0.685       # vacuum energy density

rho_c = 2.78e11 *h*h    # in units of M_s/Mpc^3

# mean matter density of the universe
rho_0 = Om_m * rho_c    # in units of M_s/Mpc^3

X_H = 0.75 #Fraction of hydrogen

@jit(nopython=True)
def H(z): #in km/yr*km
    return (6.85 * 10 ** (-11)) * np.sqrt( Om_m*(1+z)**3 + Om_lambda )

### Press-Schechter Formalism

delta_c = 1.686         # Critical density for spherical collapse

# for normalising the power spectrum
sigma8_obs = 0.811      # Observed value of sigma_8
n = 0.965               # power spectrum P(k)~ k^n

### Alternate Cosmology used in Current X-ray Background work

Om_m_new = 0.27
Om_lambda_new = 0.73
h_new = 0.7