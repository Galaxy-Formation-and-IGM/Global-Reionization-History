import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp

h = 0.67 

Om_m = 0.315            # matter density
Om_b = 0.022/(h*h)      # baryonic matter density
Om_lambda = 0.685       # vacuum energy density

rho_c = 2.78e11 *h*h    # in units of M_s/Mpc^3

# mean matter density of the universe
rho_0 = Om_m * rho_c    # in units of M_s/Mpc^3
delta_c = 1.686         # Critical density for spherical collapse

# for normalising the power spectrum
sigma8_obs = 0.811      # Observed value of sigma_8
n = 0.965               # power spectrum P(k)~ k^n

z_range = np.linspace(0.1,20, 100)

from numba import jit

f_b = Om_b/Om_m     # fraction of baryonic matter
H0 = h/9.78e9         # in units of 1/yr

@jit(nopython=True)
def H(z): #in km/yr*km
    return H0 * np.sqrt( Om_m*(1+z)**3 + Om_lambda )

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
    return k**n

# Transfer Function
def T(k):
    Gamma = Om_m * h * np.exp( -Om_b* (1+np.sqrt(2*h)/Om_m) )
    q = k / (Gamma*h)
    value = 1 + (3.89*q) + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4
    return np.log(1+2.34*q) / (2.34*q*value**(1/4))

def D(z):
    Dz = (Om_m + 0.4545*Om_lambda) / (Om_m*(1+z)**3 + 0.4545*Om_lambda)
    return Dz**(1/3)

# Variance of smoothed density field (wrt radius, R)
def sigma_R(R,z=0,cutoff=1000):    
    # Integrand for the variance calculation
    def integral(k):
        return P(k,z,cutoff) * k**2 * W(k,R)**2 / (2 * np.pi**2)
    
    sigma2, _ = quad(integral, 0, np.inf, limit=500, epsabs=1e-5, epsrel=1e-5)
    return np.sqrt(sigma2)


# Variance of smoothed density field (wrt mass of region, M)
def sigma_M(M,z=0,cutoff=1000):
    # Compute R corresponding to the mass M
    R = (3 * M / (4 * np.pi * rho_0))**(1/3)
    
    return sigma_R(R,z,cutoff)

# Evaluate and print the normalization factor
R8 = 8/h      # units of Mpc

A = sigma8_obs/sigma_R(R8,0)

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

R8 = 8/h      # units of Mpc

def sigma_R(R,z=0,cutoff=1000):    
    # Integrand for the variance calculation
    def integral(k):
        return P(k,z,cutoff) * k**2 * W(k,R)**2 / (2 * np.pi**2)
    
    sigma2, _ = quad(integral, 0, np.inf, limit=500, epsabs=1e-5, epsrel=1e-5)
    return np.sqrt(sigma2)

def sigma_M(M,z=0,cutoff=1000):
    # Compute R corresponding to the mass M
    R = (3 * M / (4 * np.pi * rho_0))**(1/3)
    
    return sigma_R(R,z,cutoff)

A = sigma8_obs/sigma_R(R8,0)

sigma_range = [A*sigma_M(M) for M in M_range]

cof = poly_fit(logM,sigma_range,degree=9)
d_cof = np.polyder(cof)

fit_deriv = np.poly1d(d_cof)
    
def PS_MassFunc(M, z):
    # Generate the fitted polynomial function
    polynomial = np.poly1d(cof)
    sigma0 = polynomial(np.log(M))
    d_sigma = abs(fit_deriv(np.log(M)))
    factor = np.sqrt(2 / np.pi) * delta_c/(sigma0**2 * D(z)) * abs(fit_deriv(np.log(M)))
    exponent = (-delta_c**2 / (2 * sigma0**2 * D(z)**2))
    return factor * np.exp(exponent)

def Madau(z):
    return 0.01 * (1+z)**2.6 / ( 1+((1+z)/3.2)**6.2 )

def Harikane(z):
    return 1 / ( 61.7 * (1+z)**(-3.13) + 10**(0.22*(1+z)) + 2.4 * 10**(0.5*(1+z)-3) )

def New_SFR(z): #Khaire
    return 10 ** (-2) * (2.01 + 8.48 * z) / (1 + (z / 2.5) ** 3.09)

mean_cx = np.linspace(2.6 * 10 ** (39), 3.7 * 10 ** (39)) #erg s^-1 / M yr^-1, Dijkstra (2012)
mean_cx_scat = mean_cx * np.e ** (1 / 2 * (0.4 * np.log(10)) ** 2)

Om_m_new = 0.27
Om_lambda_new = 0.73
h_new = 0.7

def Hopkin_SFRD(z): #Hopkin and Beacom (2006)
    a = 0.017
    b = 0.13
    c = 3.3
    d = 5.3
    return (a + b * z) * h_new / (1 + (z / c) ** d) #in M yr^-1 cMpc^-3

n_gamma = 4800 #Number of ionizing photons produced per baryon (proton), dimensionless
n_gamma2 = 7780
f_esc = 0.1 #Fraction of photons escaping the star forming halo
f_esc2 = 0.2
m_p = 1.67 * 10 ** (-27) * 5.03 * 10 ** (-31) #mass of proton in Solar Masses

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

X_H = 0.75 #Fraction of hydrogen
alpha_B = 2.59 * 10 ** (-13) #Case B recombination coefficient at T=3*10^4 K in cm^3 * s^-1

alpha_B_unit = alpha_B * (3.24 * 10 ** (-25)) ** 3 / (3.17 * 10 ** (-8)) #Coefficient in Mpc^3 * yr^-1

def n_H(z): #Mean proper number denisty of hydrogen atoms in Mpc^-3
    return X_H * Om_b * rho_c * (1 + z) ** 3 / m_p

def dt_dz(z): #In yr
    return -1 / ((1 + z) * H(z))

def dN_dt_Mad(z):
    return Madau(z) * (1 + z) ** 3 * n_gamma2 * f_esc / m_p

def Clump(z):
    #if z >= 6:
     #   return 1 + 9 * (7 / (1 + z)) ** 2
    #else:
     #   return 10
    #return 3
    return 2.9 * ((1 + z) / 6) ** (-1.1)

def RHS_Mad(z, f_HII):
    return dN_dt_Mad(z) * dt_dz(z) / n_H(z) - alpha_B_unit * n_H(z) * f_HII * Clump(z) * dt_dz(z)

Mad_data = RK4(RHS_Mad, 20, 0, -0.1, 0.01)

"""
Notebook: Alternat_XRaySources
Folder: SFRD_Probes
"""

def E_bb(M, z): #Baryon binding energy of a halo of mass M (in SM) at redshift z in erg
    Om_m_z = Om_m * (1 + z) ** 3 / (Om_m * (1 + z) ** 3 + Om_lambda)
    d = Om_m_z - 1
    delta_c = 18 * np.pi ** 2 + 82 * d - 39 * d ** 2
    baryon_frac = Om_b / Om_m #Need only baryon binding energy
    return 5.45 * 10 ** (53) * (M * h / 10 ** 8) ** (5 / 3) * (Om_m / Om_m_z * delta_c / (18 * np.pi ** 2)) ** (1 / 3) * (1 + z) / (10 * h) * baryon_frac

min_T_vir = 10 ** 4 #Efficient atomic cooling above this temp

neutral_mu = 1.23
k_B_SI = 1.38 * 10 ** (-23) #m^2 kg s^-2 K^-1
m_p_kg = 1.67 * 10 ** (-27) #kg

min_V_c = np.sqrt(min_T_vir * 2 * (k_B_SI / (1000) ** 2) / (neutral_mu * m_p_kg))

def M_factor(z): #The coefficient of M^1/3 in V_c formula
    Om_m_z = Om_m * (1 + z) ** 3 / (Om_m * (1 + z) ** 3 + Om_lambda)
    d = Om_m_z - 1
    delta_c = 18 * np.pi ** 2 + 82 * d - 39 * d ** 2
    return 23.4 * (Om_m / Om_m_z * delta_c / (18 * np.pi ** 2)) ** (1 / 6) * ((1 + z) / 10) ** (1 / 2) / (10 ** 8 * h ** (-1)) ** (1 / 3)

def M_crit(z):
    return (min_V_c / M_factor(z)) ** (3)

M_range = np.logspace(6, 16, 500)  # Mass range in solar masses

M_arrs_inz = [] #Mass range set by V_c critical

for i in range(len(z_range)):
    arr = np.logspace(np.log10(M_crit(z_range[i])), 16, 500)
    M_arrs_inz.append(arr)

M_arrs_inz = np.array(M_arrs_inz)

dn_dlogM_new = np.zeros((len(M_range), len(z_range)))
integrand_rel_en_dens = np.zeros((len(M_range), len(z_range)))

f_r = 0.1

Rel_energy_dens = np.zeros(len(z_range))

for i in range(len(z_range)):

    for j in range(len(M_range)):
        M = M_arrs_inz[i][j]

        dn_dlogM_new[j, i] = PS_MassFunc(M, z_range[i]) * (rho_0 / M)
        integrand_rel_en_dens[j, i] = (
            dn_dlogM_new[j, i]
            * f_r
            * E_bb(M, z_range[i])
        )

    Rel_energy_dens[i] = np.trapezoid(
        integrand_rel_en_dens[:, i],
        x=np.log10(M_arrs_inz[i])
    )

Yrtosec = 3.154 * 10 ** 7

def Time_ff(z): #H(z) is in yr^{-1}
    return 1 / np.sqrt(27 * np.pi ** 2 * (H(z) / Yrtosec) ** 2)

Time_ff_vals = Time_ff(z_range)

AccShocks_Em_vals = Rel_energy_dens / Time_ff_vals

###High Mass Xray Binary Emissivity:

SFRD_vals = np.zeros_like(z_range)
HMXB_Em_vals = np.zeros_like(z_range)

L_HMXB = 10 ** (40)

for i in range(len(z_range)):
    SFRD_vals[i] = Madau(z_range[i])
    HMXB_Em_vals[i] = L_HMXB * SFRD_vals[i]

### Low Mass Xray Binary Emissivity:

def dt_dz(z): #In yr
    return -1 / ((1 + z) * H(z))

def LM_Int(z):
    return Madau(z) * np.abs(dt_dz(z))

LM_Int_vals = np.zeros_like(z_range)

for i in range(len(z_range)):
    LM_Int_vals[i] = LM_Int(z_range[i])

Madau_vals = Madau(z_range)

from scipy.integrate import cumulative_trapezoid

z = z_range                # ascending: 0 → 20
f = LM_Int(z)

# cumulative integral from 0 → z
cum_int = cumulative_trapezoid(f, z, initial=0)

# total integral from 0 → 20
total = cum_int[-1]

# desired: integral from z → 20
F = total - cum_int

L_LMXB = 10 ** (30.5)

Ret_fact = 0.39

LMXB_Em_vals = L_LMXB * F * (1 - Ret_fact)

### Varying f_R

f_r_2 = 0.05
f_r_3 = 0.01

dn_dlogM_new_2 = np.zeros((len(M_range), len(z_range)))
integrand_rel_en_dens_2 = np.zeros((len(M_range), len(z_range)))

for i in range(len(z_range)):
    for j in range(len(M_range)):
        dn_dlogM_new_2[j, i] = PS_MassFunc(M_arrs_inz[i][j], z_range[i]) * (rho_0 / M_arrs_inz[i][j])
        integrand_rel_en_dens_2[j, i] = dn_dlogM_new_2[j, i] * f_r_2 * E_bb(M_arrs_inz[i][j], z_range[i])

dn_dlogM_new_3 = np.zeros((len(M_range), len(z_range)))
integrand_rel_en_dens_3 = np.zeros((len(M_range), len(z_range)))

for i in range(len(z_range)):
    for j in range(len(M_range)):
        dn_dlogM_new_3[j, i] = PS_MassFunc(M_arrs_inz[i][j], z_range[i]) * (rho_0 / M_arrs_inz[i][j])
        integrand_rel_en_dens_3[j, i] = dn_dlogM_new_3[j, i] * f_r_3 * E_bb(M_arrs_inz[i][j], z_range[i])

Rel_energy_dens_2 = np.trapezoid(integrand_rel_en_dens_2, x=np.log10(M_range), axis=0)
Rel_energy_dens_3 = np.trapezoid(integrand_rel_en_dens_3, x=np.log10(M_range), axis=0)

AccShocks_Em_vals_2 = Rel_energy_dens_2 / Time_ff_vals
AccShocks_Em_vals_3 = Rel_energy_dens_3 / Time_ff_vals