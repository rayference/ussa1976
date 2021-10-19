"""
Constants module.

As much as possible, constants' names are chosen to be as close as possible to
the notations used in :cite:`NASA1976USStandardAtmosphere`.
"""
import numpy as np

from .units import ureg

# Boltzmann constant
K = 1.380622e-23 * ureg.joule / ureg.kelvin

# Molar masses of the individual species
M = {
    "N2": 0.0280134 * ureg.kg / ureg.mole,
    "O2": 0.0319988 * ureg.kg / ureg.mole,
    "Ar": 0.039948 * ureg.kg / ureg.mole,
    "CO2": 0.04400995 * ureg.kg / ureg.mole,
    "Ne": 0.020183 * ureg.kg / ureg.mole,
    "He": 0.0040026 * ureg.kg / ureg.mole,
    "Kr": 0.08380 * ureg.kg / ureg.mole,
    "Xe": 0.13130 * ureg.kg / ureg.mole,
    "CH4": 0.01604303 * ureg.kg / ureg.mole,
    "H2": 0.00201594 * ureg.kg / ureg.mole,
    "O": 0.01599939 * ureg.kg / ureg.mole,
    "H": 0.00100797 * ureg.kg / ureg.mole,
}

# Sea level mean (mixture) molar mass
M0 = 0.028964425278793997 * ureg.kilogram / ureg.mole

# Avogadro number
NA = 6.022169e23 / ureg.mole

# Universal gas constant
R = 8.31432 * ureg.joule / (ureg.mole * ureg.kelvin)

# Sea level volume fractions of the gas species present below 86 km
F = {
    "N2": 0.78084 * ureg.dimensionless,
    "O2": 0.209476 * ureg.dimensionless,
    "Ar": 0.00934 * ureg.dimensionless,
    "CO2": 0.000314 * ureg.dimensionless,
    "Ne": 0.00001818 * ureg.dimensionless,
    "He": 0.00000524 * ureg.dimensionless,
    "Kr": 0.00000114 * ureg.dimensionless,
    "Xe": 0.000000087 * ureg.dimensionless,
    "CH4": 0.000002 * ureg.dimensionless,
    "H2": 0.0000005 * ureg.dimensionless,
}

# Sea level gravity
G0 = 9.80665 * ureg.m / ureg.s ** 2

# Geopotential altitudes of the layers' boundaries (below 86 km)
H = (
    np.array(
        [
            0.0,
            11e3,
            20e3,
            32e3,
            47e3,
            51e3,
            71e3,
            84852.05,
        ]
    )
    * ureg.m
)

# Temperature gradients in the seven layers (below 86 km)
LK = (
    np.array(
        [
            -0.0065,
            0.0,
            0.0010,
            0.0028,
            0.0,
            -0.0028,
            -0.0020,
        ]
    )
    * ureg.kelvin
    / ureg.meter
)

# Pressure at sea level
P0 = 101325.0 * ureg.pascal

# Effective Earth radius
R0 = 6.356766e6 * ureg.meter

# Temperature at sea level
T0 = 288.15 * ureg.kelvin
S = 110.4 * ureg.kelvin
BETA = 1.458e6 * ureg.kilogram / (ureg.meter * ureg.second * ureg.kelvin ** 0.5)
GAMMA = 1.40 * ureg.dimensionless
SIGMA = 3.65e-10 * ureg.meter

# Thermal diffusion constants of the individual species present above 86 km
ALPHA = {
    "N2": 0.0 * ureg.dimensionless,
    "O": 0.0 * ureg.dimensionless,
    "O2": 0.0 * ureg.dimensionless,
    "Ar": 0.0 * ureg.dimensionless,
    "He": -0.4 * ureg.dimensionless,
    "H": -0.25 * ureg.dimensionless,
}
A = {
    "N2": None,
    "O": 6.986e20 / ureg.meter / ureg.second,
    "O2": 4.863e20 / ureg.meter / ureg.second,
    "Ar": 4.487e20 / ureg.meter / ureg.second,
    "He": 1.7e21 / ureg.meter / ureg.second,
    "H": 3.305e21 / ureg.meter / ureg.second,
}
B = {
    "N2": None,
    "O": 0.75 * ureg.dimensionless,
    "O2": 0.75 * ureg.dimensionless,
    "Ar": 0.87 * ureg.dimensionless,
    "He": 0.691 * ureg.dimensionless,
    "H": 0.5 * ureg.dimensionless,
}

# Eddy diffusion coefficients
K_7 = 1.2e2 * ureg.meter ** 2 / ureg.second

# Vertical transport constants of the individual species present above 86 km
Q1 = {
    "O": -5.809644e-4 / ureg.kilometer ** 3,
    "O2": 1.366212e-4 / ureg.kilometer ** 3,
    "Ar": 9.434079e-5 / ureg.kilometer ** 3,
    "He": -2.457369e-4 / ureg.kilometer ** 3,
}
Q2 = {
    "O": -3.416248e-3 / ureg.kilometer ** 3,  # warning: above 97 km, Q2 = 0.
    "O2": 0.0 / ureg.kilometer ** 3,
    "Ar": 0.0 / ureg.kilometer ** 3,
    "He": 0.0 / ureg.kilometer ** 3,
}
U1 = {
    "O": 56.90311 * ureg.kilometer,
    "O2": 86.0 * ureg.kilometer,
    "Ar": 86.0 * ureg.kilometer,
    "He": 86.0 * ureg.kilometer,
}
U2 = {"O": 97.0 * ureg.kilometer, "O2": None, "Ar": None, "He": None}
W1 = {
    "O": 2.706240e-5 / ureg.kilometer ** 3,
    "O2": 8.333333e-5 / ureg.kilometer ** 3,
    "Ar": 8.333333e-5 / ureg.kilometer ** 3,
    "He": 6.666667e-4 / ureg.kilometer ** 3,
}
W2 = {"O": 5.008765e-4 / ureg.kilometer ** 3, "O2": None, "Ar": None, "He": None}

# Altitudes of the levels delimiting 5 layers above 86 km
Z7 = 86.0 * ureg.kilometer
Z8 = 91.0 * ureg.kilometer
Z9 = 110.0 * ureg.kilometer
Z10 = 120.0 * ureg.kilometer
Z12 = 1000.0 * ureg.kilometer

# Temperature at the different levels above 86 km
T7 = 186.8673 * ureg.kelvin
T9 = 240.0 * ureg.kelvin
T10 = 360.0 * ureg.kelvin
T11 = 999.2356 * ureg.kelvin
TINF = 1000.0 * ureg.kelvin
LAMBDA = 0.01875 / ureg.kilometer

# Temperature gradients
LK7 = 0.0 * ureg.kelvin / ureg.kilometer
LK9 = 12.0 * ureg.kelvin / ureg.kilometer

# Molecular nitrogen at altitude = Z7
N2_7 = 1.129794e20 / ureg.meter ** 3

# Atomic oxygen at altitude = Z7
O_7 = 8.6e16 / ureg.meter ** 3

# Molecular oxygen at altitude = Z7
O2_7 = 3.030898e19 / ureg.meter ** 3

# Argon at altitude = Z7
AR_7 = 1.351400e18 / ureg.meter ** 3

# Helium at altitude = Z7 (assumes typo at page 13)
HE_7 = 7.5817e14 / ureg.meter ** 3

# Hydrogen at altitude = Z7
H_11 = 8.0e10 / ureg.meter ** 3

# Vertical flux
PHI = 7.2e11 / ureg.meter ** 2 / ureg.second
