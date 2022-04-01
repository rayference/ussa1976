"""Constants module.

As much as possible, constants' names are chosen to be as close as possible to
the notations used in :cite:`NASA1976USStandardAtmosphere`.

Constants' values are evaluated in the following set of units:

    * length: meter
    * time: second
    * mass: kilogram
    * temperature: kelvin
    * quantity of matter: mole

Note the following derived units:

    * 1 Pa = 1 kg * m^-1 * s^-2
    * 1 Joule = 1 kg * m^2 * s^-2
"""
import numpy as np
import numpy.typing as npt

# Boltzmann constant
K = 1.380622e-23  # J * K^-1

# Molar masses of the individual species
M = {
    "N2": 0.0280134,  # kg * mole^-1
    "O2": 0.0319988,  # kg * mole^-1
    "Ar": 0.039948,  # kg * mole^-1
    "CO2": 0.04400995,  # kg * mole^-1
    "Ne": 0.020183,  # kg * mole^-1
    "He": 0.0040026,  # kg * mole^-1
    "Kr": 0.08380,  # kg * mole^-1
    "Xe": 0.13130,  # kg * mole^-1
    "CH4": 0.01604303,  # kg * mole^-1
    "H2": 0.00201594,  # kg * mole^-1
    "O": 0.01599939,  # kg * mole^-1
    "H": 0.00100797,  # kg * mole^-1
}

# Sea level mean (mixture) molar mass
M0 = 0.028964425278793997  # kg * mole^-1

# Avogadro number
NA = 6.022169e23  # mole^-1

# Universal gas constant
R = 8.31432  # J * K^-1 * mole^-1

# Sea level volume fractions of the gas species present below 86 km
F = {
    "N2": 0.78084,  # dimensionless
    "O2": 0.209476,  # dimensionless
    "Ar": 0.00934,  # dimensionless
    "CO2": 0.000314,  # dimensionless
    "Ne": 0.00001818,  # dimensionless
    "He": 0.00000524,  # dimensionless
    "Kr": 0.00000114,  # dimensionless
    "Xe": 0.000000087,  # dimensionless
    "CH4": 0.000002,  # dimensionless
    "H2": 0.0000005,  # dimensionless
}

# Sea level gravity
G0 = 9.80665  # m / s^-2

# Geopotential altitudes of the layers' boundaries (below 86 km)
H: npt.NDArray[np.float64] = np.array(
    [
        0.0,
        11e3,
        20e3,
        32e3,
        47e3,
        51e3,
        71e3,
        84852.05,
    ]  # m
)

# Temperature gradients in the seven layers (below 86 km)
LK: npt.NDArray[np.float64] = np.array(
    [
        -0.0065,
        0.0,
        0.0010,
        0.0028,
        0.0,
        -0.0028,
        -0.0020,
    ]  # K * m^-1
)

# Pressure at sea level
P0 = 101325.0  # Pa

# Effective Earth radius
R0 = 6.356766e6  # m

# Temperature at sea level
T0 = 288.15  # K
S = 110.4  # K
BETA = 1.458e6  # kg * m^-1 * s^-1 * K^-0.5
GAMMA = 1.40  # dimensionless
SIGMA = 3.65e-10  # m

# Thermal diffusion constants of the individual species present above 86 km
ALPHA = {
    "N2": 0.0,  # dimensionless
    "O": 0.0,  # dimensionless
    "O2": 0.0,  # dimensionless
    "Ar": 0.0,  # dimensionless
    "He": -0.4,  # dimensionless
    "H": -0.25,  # dimensionless
}
A = {
    # "N2": None,
    "O": 6.986e20,  # m * s^-1
    "O2": 4.863e20,  # m * s^-1
    "Ar": 4.487e20,  # m * s^-1
    "He": 1.7e21,  # m * s^-1
    "H": 3.305e21,  # m * s^-1
}
B = {
    # "N2": None,
    "O": 0.75,  # dimensionless
    "O2": 0.75,  # dimensionless
    "Ar": 0.87,  # dimensionless
    "He": 0.691,  # dimensionless
    "H": 0.5,  # dimensionless
}

# Eddy diffusion coefficients
K_7 = 1.2e2  # m^2 * s^-1

# Vertical transport constants of the individual species present above 86 km
Q1 = {
    "O": -5.809644e-13,  # m^-3
    "O2": 1.366212e-13,  # m^-3
    "Ar": 9.434079e-14,  # m^-3
    "He": -2.457369e-13,  # m^-3
}
Q2 = {
    "O": -3.416248e-12,  # m^-3  # warning: above 97 km, Q2 = 0 m^-3.
    "O2": 0.0,  # m^-3
    "Ar": 0.0,  # m^-3
    "He": 0.0,  # m^-3
}
U1 = {
    "O": 56.90311e3,  # m
    "O2": 86e3,  # m
    "Ar": 86e3,  # m
    "He": 86e3,  # m
}
U2 = {"O": 97e3}  # "O2": None, "Ar": None, "He": None}  # m
W1 = {
    "O": 2.706240e-14,  # m^-3
    "O2": 8.333333e-14,  # m^-3
    "Ar": 8.333333e-14,  # m^-3
    "He": 6.666667e-13,  # m^-3
}
W2 = {
    "O": 5.008765e-13,  # m^-3
    #    "O2": None,
    #    "Ar": None,
    #    "He": None,
}

# Altitudes of the levels delimiting 5 layers above 86 km
Z7 = 86e3  # m
Z8 = 91e3  # m
Z9 = 110e3  # m
Z10 = 120e3  # m
Z12 = 1000e3  # m

# Temperature at the different levels above 86 km
T7 = 186.8673  # K
T9 = 240.0  # K
T10 = 360.0  # K
T11 = 999.2356  # K
TINF = 1000.0  # K
LAMBDA = 0.01875e-3  # m^-1

# Temperature gradients
LK7 = 0.0  # K * m^-1
LK9 = 12.0e-3  # K * m^-1

# Molecular nitrogen at altitude = Z7
N2_7 = 1.129794e20  # m^-3

# Atomic oxygen at altitude = Z7
O_7 = 8.6e16  # m^-3

# Molecular oxygen at altitude = Z7
O2_7 = 3.030898e19  # m^-3

# Argon at altitude = Z7
AR_7 = 1.351400e18  # m^-3

# Helium at altitude = Z7 (assumes typo at page 13)
HE_7 = 7.5817e14  # m^-3

# Hydrogen at altitude = Z7
H_11 = 8.0e10  # m^-3

# Vertical flux
PHI = 7.2e11  # m^2 * s^-1
