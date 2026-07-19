### Basic package import

import numpy as np
from scipy.integrate import cumulative_trapezoid

### src imports

import Cosmology as cosmo
import Constants_Units as cu
from Theoretical_Setup import z_range, PS_MassFunc
from Cosmological_Reionization import Madau

### Generating Accretion shock emissivity

def E_bb(M, z): #Baryon binding energy of a halo of mass M (in SM) at redshift z in erg
    Om_m_z = cosmo.Om_m * (1 + z) ** 3 / (cosmo.Om_m * (1 + z) ** 3 + cosmo.Om_lambda)
    d = Om_m_z - 1
    DELTA_c = 18 * np.pi ** 2 + 82 * d - 39 * d ** 2
    baryon_frac = cosmo.Om_b / cosmo.Om_m #Need only baryon binding energy
    return 5.45 * 10 ** (53) * (M * cosmo.h / 10 ** 8) ** (5 / 3) * (cosmo.Om_m / Om_m_z * DELTA_c / (18 * np.pi ** 2)) ** (1 / 3) * (1 + z) / (10 * cosmo.h) * baryon_frac

min_T_vir = 10 ** 4 #Efficient atomic cooling above this temp

neutral_mu = 1.23

min_V_c = np.sqrt(min_T_vir * 2 * (cu.k_B / (1000) ** 2) / (neutral_mu * cu.m_p))

def M_factor(z): #The coefficient of M^1/3 in V_c formula
    Om_m_z = cosmo.Om_m * (1 + z) ** 3 / (cosmo.Om_m * (1 + z) ** 3 + cosmo.Om_lambda)
    d = Om_m_z - 1
    DELTA_c = 18 * np.pi ** 2 + 82 * d - 39 * d ** 2
    return 23.4 * (cosmo.Om_m / Om_m_z * DELTA_c / (18 * np.pi ** 2)) ** (1 / 6) * ((1 + z) / 10) ** (1 / 2) / (10 ** 8 * cosmo.h ** (-1)) ** (1 / 3)

def M_crit(z):
    return (min_V_c / M_factor(z)) ** (3)

M_arrs_inz = [] #Mass range set by V_c critical

for i in range(len(z_range)):
    arr = np.logspace(np.log10(M_crit(z_range[i])), 16, 500)
    M_arrs_inz.append(arr)

M_arrs_inz = np.array(M_arrs_inz)

dn_dlogM_new = np.zeros((len(M_arrs_inz[0]), len(z_range)))
integrand_rel_en_dens = np.zeros((len(M_arrs_inz[0]), len(z_range)))

f_r = 0.1

Rel_energy_dens = np.zeros(len(z_range))

for i in range(len(z_range)):

    for j in range(len(M_arrs_inz[i])):
        M = M_arrs_inz[i][j]

        dn_dlogM_new[j, i] = PS_MassFunc(M, z_range[i]) * (cosmo.rho_0 / M)
        integrand_rel_en_dens[j, i] = (
            dn_dlogM_new[j, i]
            * f_r
            * E_bb(M, z_range[i])
        )

    Rel_energy_dens[i] = np.trapezoid(
        integrand_rel_en_dens[:, i],
        x=np.log10(M_arrs_inz[i])
    )

def Time_ff(z): #H(z) is in yr^{-1}
    return 1 / np.sqrt(27 * np.pi ** 2 * (cosmo.H(z) / cu.YrTos) ** 2)

Time_ff_vals = Time_ff(z_range)

AccShocks_Em_vals = Rel_energy_dens / Time_ff_vals

###High Mass Xray Binary Emissivity:

SFRD_vals = Madau(z_range)
HMXB_Em_vals = np.zeros_like(z_range)

L_HMXB = 10 ** (40)

for i in range(len(z_range)):
    HMXB_Em_vals[i] = L_HMXB * SFRD_vals[i]

### Low Mass Xray Binary Emissivity:

def dt_dz(z): #In yr
    return -1 / ((1 + z) * cosmo.H(z))

def LM_Int(z):
    return Madau(z) * np.abs(dt_dz(z))

Madau_vals = Madau(z_range)

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

dn_dlogM_new_2 = np.zeros((len(M_arrs_inz[0]), len(z_range)))
integrand_rel_en_dens_2 = np.zeros((len(M_arrs_inz[0]), len(z_range)))

for i in range(len(z_range)):
    for j in range(len(M_arrs_inz[i])):
        dn_dlogM_new_2[j, i] = PS_MassFunc(M_arrs_inz[i][j], z_range[i]) * (cosmo.rho_0 / M_arrs_inz[i][j])
        integrand_rel_en_dens_2[j, i] = dn_dlogM_new_2[j, i] * f_r_2 * E_bb(M_arrs_inz[i][j], z_range[i])

dn_dlogM_new_3 = np.zeros((len(M_arrs_inz[0]), len(z_range)))
integrand_rel_en_dens_3 = np.zeros((len(M_arrs_inz[0]), len(z_range)))

for i in range(len(z_range)):
    for j in range(len(M_arrs_inz[i])):
        dn_dlogM_new_3[j, i] = PS_MassFunc(M_arrs_inz[i][j], z_range[i]) * (cosmo.rho_0 / M_arrs_inz[i][j])
        integrand_rel_en_dens_3[j, i] = dn_dlogM_new_3[j, i] * f_r_3 * E_bb(M_arrs_inz[i][j], z_range[i])

Rel_energy_dens_2 = np.zeros(len(z_range))

for i in range(len(z_range)):
    Rel_energy_dens_2[i] = np.trapezoid(
        integrand_rel_en_dens_2[:, i],
        x=np.log10(M_arrs_inz[i])
    )

Rel_energy_dens_3 = np.zeros(len(z_range))

for i in range(len(z_range)):
    Rel_energy_dens_3[i] = np.trapezoid(
        integrand_rel_en_dens_3[:, i],
        x=np.log10(M_arrs_inz[i])
    )

AccShocks_Em_vals_2 = Rel_energy_dens_2 / Time_ff_vals
AccShocks_Em_vals_3 = Rel_energy_dens_3 / Time_ff_vals