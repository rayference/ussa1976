"""
U.S. Standard Atmosphere 1976 thermophysical model.

U.S. Standard Atmosphere, 1976 thermophysical model according to
:cite:`NASA1976USStandardAtmosphere`.
"""
import datetime
import typing as t

import numpy as np
import numpy.ma as ma
import numpy.typing as npt
import pint
import xarray as xr
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from . import __version__
from .units import to_quantity
from .units import ureg

# ------------------------------------------------------------------------------
#
# Atmospheric vertical profile data set generator
#
# ------------------------------------------------------------------------------

_DEFAULT_LEVELS = ureg.Quantity(np.linspace(0.0, 100.0, 51), "km")


def make(levels: pint.Quantity = _DEFAULT_LEVELS) -> xr.Dataset:
    """Make U.S. Standard Atmosphere 1976.

    Parameters
    ----------
    levels: quantity, optional
        Level altitudes.
        The values must be sorted by increasing order.
        The default levels are 51 linearly spaced values between 0 and 100 km.

        Valid range: 0 to 1000 km.

    Returns
    -------
    Dataset
        Data set holding the values of the pressure, temperature,
        total number density and number densities of the individual
        gas species in each layer.

    Raises
    ------
    ValueError
        When levels are out of range.

    Notes
    -----
    The pressure, temperature and number densities given in each layer of
    the altitude mesh are computed at the altitude of the layers centers.
    In other words, the layer's middle is taken as the altitude
    representative of the whole layer. For example, in a layer with lower
    and upper altitudes of 1000 and 2000 m, the thermophysical variables
    are computed at the altitude of 1500 m.
    """
    if np.any(levels > ureg.Quantity(1e6, "m")) or np.any(levels < 0.0):
        raise ValueError("Levels altitudes must be in [0, 1e6] m.")

    z_layer = (levels[:-1] + levels[1:]) / 2

    # create the data set
    ds = create(
        z=z_layer,
        variables=["p", "t", "n", "n_tot"],
    )

    # derive atmospheric thermophysical properties profile data set
    thermoprops_ds = (
        xr.Dataset(
            data_vars={
                "p": ds.p,
                "t": ds.t,
                "n": ds.n_tot,
                "mr": ds.n / ds.n_tot,
            }
        )
        .rename_dims({"z": "z_layer"})
        .reset_coords("z", drop=True)
    )
    thermoprops_ds.coords["z_layer"] = (
        "z_layer",
        z_layer.magnitude,
        dict(
            standard_name="layer_altitude",
            long_name="layer altitude",
            units=str(z_layer.units),
        ),
    )
    thermoprops_ds.coords["z_level"] = (
        "z_level",
        levels.magnitude,
        dict(
            standard_name="level_altitude",
            long_name="level altitude",
            units=str(levels.units),
        ),
    )
    thermoprops_ds.attrs = dict(
        convention="CF-1.8",
        title="U.S. Standard Atmosphere 1976",
        history=f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} - "
        f"data creation - ussa1976",
        source=f"ussa1976, version {__version__}",
        references="U.S. Standard Atmosphere, 1976, NASA-TM-X-74335, "
        "NOAA-S/T-76-1562",
    )

    return thermoprops_ds


# ------------------------------------------------------------------------------
#
# Constants.
# As much as possible, constants names are chosen to be as close as possible to
# the notations used in :cite:`NASA1976USStandardAtmosphere`.
#
# ------------------------------------------------------------------------------

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
    "O": 6.986e20 / ureg.meter / ureg.second,  # [m^-1*s^-1]
    "O2": 4.863e20 / ureg.meter / ureg.second,  # [m^-1*s^-1]
    "Ar": 4.487e20 / ureg.meter / ureg.second,  # [m^-1*s^-1]
    "He": 1.7e21 / ureg.meter / ureg.second,  # [m^-1*s^-1]
    "H": 3.305e21 / ureg.meter / ureg.second,  # [m^-1*s^-1]
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
K_10 = 0.0 * ureg.meter ** 2 / ureg.second

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
Z11 = 500.0 * ureg.kilometer
Z12 = 1000.0 * ureg.kilometer

# Temperature at the different levels above 86 km
T7 = 186.8673 * ureg.kelvin  # [K]
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

# List of all gas species
SPECIES = ["N2", "O2", "Ar", "CO2", "Ne", "He", "Kr", "Xe", "CH4", "H2", "O", "H"]

# List of variables computed by the model
VARIABLES = [
    "t",
    "p",
    "n",
    "n_tot",
    "rho",
    "mv",
    "hp",
    "v",
    "mfp",
    "f",
    "cs",
    "mu",
    "nu",
    "kt",
]

# Variables standard names with respect to the Climate and Forecast (CF)
# convention
STANDARD_NAME = {
    "t": "air_temperature",
    "p": "air_pressure",
    "n": "number_density",
    "n_tot": "air_number_density",
    "rho": "air_density",
    "mv": "molar_volume",
    "hp": "pressure_scale_height",
    "v": "mean_air_particles_speed",
    "mfp": "mean_free_path",
    "f": "mean_collision_frequency",
    "cs": "speed_of_sound_in_air",
    "mu": "air_dynamic_viscosity",
    "nu": "air_kinematic_viscosity",
    "kt": "air_thermal_conductivity_coefficient",
    "z": "altitude",
    "h": "geopotential_height",
}

# Units of relevant quantities
UNITS = {
    "t": "K",
    "p": "Pa",
    "n": "m^-3",
    "n_tot": "m^-3",
    "rho": "kg/m^3",
    "mv": "m^3/mole",
    "hp": "m",
    "v": "m/s",
    "mfp": "m",
    "f": "s^-1",
    "cs": "m/s",
    "mu": "kg/(m*s)",
    "nu": "m^2/s",
    "kt": "W/(m*K)",
    "z": "m",
    "h": "m",
    "species": "",
}

# Variables dimensions
DIMS = {
    "t": "z",
    "p": "z",
    "n": ("species", "z"),
    "n_tot": "z",
    "rho": "z",
    "mv": "z",
    "hp": "z",
    "v": "z",
    "mfp": "z",
    "f": "z",
    "cs": "z",
    "mu": "z",
    "nu": "z",
    "kt": "z",
}


# ------------------------------------------------------------------------------
#
# Computational functions.
# The U.S. Standard Atmosphere 1976 model divides the atmosphere into two
# altitude regions:
#   1. the low-altitude region, from 0 to 86 kilometers
#   2. the high-altitude region, from 86 to 1000 kilometers.
# The majority of computational functions hereafter are specialised for one or
# the other altitude region and is valid only in that altitude region, not in
# the other.
#
# ------------------------------------------------------------------------------


def create(
    z: pint.Quantity,
    variables: t.Optional[t.List[str]] = None,
) -> xr.Dataset:
    """Create U.S. Standard Atmosphere 1976 data set.

    Parameters
    ----------
    z: quantity
        Altitude mesh.

    variables: list, optional
        Names of the variables to compute.

    Returns
    -------
    Dataset
        Data set holding the values of the different atmospheric variables.

    Raises
    ------
    ValueError
        When altitude is out of bounds, or when variables are invalid.
    """
    if np.any(z.magnitude < 0.0):
        raise ValueError("altitude values must be greater than or equal to " "zero")

    if np.any(z > ureg.Quantity(1000000.0, "m")):
        raise ValueError("altitude values must be less then or equal to 1e6 m")

    if variables is None:
        variables = VARIABLES
    else:
        for var in variables:
            if var not in VARIABLES:
                raise ValueError(var, " is not a valid variable name")

    # initialise data set
    ds = init_data_set(z=z)

    # compute the model in the low-altitude region
    z_delim = 86 * ureg.km
    low_altitude_region = ds.z <= z_delim.m_as(ds.z.units)
    compute_low_altitude(data_set=ds, mask=low_altitude_region, inplace=True)

    # compute the model in the high-altitude region
    high_altitude_region = ds.z > z_delim.m_as(ds.z.units)
    compute_high_altitude(data_set=ds, mask=high_altitude_region, inplace=True)

    # replace all np.nan with 0. in number densities values
    n = ds.n.values
    n[np.isnan(n)] = 0.0
    ds.n.values = n

    # list names of variables to drop from the data set
    names = []
    for var in ds.data_vars:  # type: ignore
        if var not in variables:
            names.append(var)

    return ds.drop_vars(names)  # type: ignore


def compute_low_altitude(
    data_set: xr.Dataset, mask: t.Optional[xr.DataArray] = None, inplace: bool = False
) -> t.Optional[xr.Dataset]:
    """Compute U.S. Standard Atmosphere 1976 in low-altitude region.

    Parameters
    ----------
    data_set: Dataset
        Data set to compute.

    mask: DataArray, optional
        Mask to select the region of the data set to compute.
        By default, the mask selects the entire data set.

    inplace: bool, default=False
        If ``True``, modifies ``data_set`` in place, else returns a copy of
        ``data_set``.

    Returns
    -------
    Dataset
        If ``inplace`` is ``True``, returns nothing, else returns a copy of
        ``data_set``.
    """
    if mask is None:
        mask = xr.full_like(data_set.coords["z"], True, dtype=bool)

    if inplace:
        ds = data_set
    else:
        ds = data_set.copy(deep=True)

    z = to_quantity(ds.z[mask])
    altitudes = z.magnitude

    # compute levels temperature and pressure values
    tb, pb = compute_levels_temperature_and_pressure_low_altitude()

    # compute geopotential height, temperature and pressure
    h = to_geopotential_height(z)
    t = compute_temperature_low_altitude(h=h, tb=tb)
    p = compute_pressure_low_altitude(h=h, pb=pb, tb=tb)

    # compute the auxiliary atmospheric variables
    n_tot = NA * p / (R * t)
    rho = p * M0 / (R * t)
    g = compute_gravity(z)
    mu = BETA * np.power(t, 1.5) / (t + S)

    # assign data set with computed values
    ds["t"].loc[dict(z=altitudes)] = t.m_as(UNITS["t"])
    ds["p"].loc[dict(z=altitudes)] = p.m_as(UNITS["p"])
    ds["n_tot"].loc[dict(z=altitudes)] = n_tot.m_as(UNITS["n_tot"])

    species = ["N2", "O2", "Ar", "CO2", "Ne", "He", "Kr", "Xe", "CH4", "H2"]
    for i, s in enumerate(SPECIES):
        if s in species:
            ds["n"][i].loc[dict(z=altitudes)] = (F[s] * n_tot).m_as(UNITS["n"])

    ds["rho"].loc[dict(z=altitudes)] = rho.m_as(UNITS["rho"])
    ds["mv"].loc[dict(z=altitudes)] = (NA / n_tot).m_as(UNITS["mv"])
    ds["hp"].loc[dict(z=altitudes)] = (R * t / (g * M0)).m_as(UNITS["hp"])
    ds["v"].loc[dict(z=altitudes)] = (np.sqrt(8.0 * R * t / (np.pi * M0))).m_as(
        UNITS["v"]
    )
    ds["mfp"].loc[dict(z=altitudes)] = (
        np.sqrt(2.0) / (2.0 * np.pi * np.power(SIGMA, 2.0) * n_tot)
    ).m_as(UNITS["mfp"])
    ds["f"].loc[dict(z=altitudes)] = (
        4.0
        * NA
        * np.power(SIGMA, 2.0)
        * np.sqrt(np.pi * np.power(p, 2.0) / (R * M0 * t))
    ).m_as(UNITS["f"])
    ds["cs"].loc[dict(z=altitudes)] = (np.sqrt(GAMMA * R * t / M0)).m_as(UNITS["cs"])
    ds["mu"].loc[dict(z=altitudes)] = mu.m_as(UNITS["mu"])
    ds["nu"].loc[dict(z=altitudes)] = (mu / rho).m_as(UNITS["nu"])
    ds["kt"].loc[dict(z=altitudes)] = (
        (
            2.64638e-3
            * np.power(t.m_as("K"), 1.5)
            / (t.m_as("K") + 245.4 * np.power(10.0, -12.0 / t.m_as("K")))
        )
        * ureg.watt
        / (ureg.meter * ureg.kelvin)
    ).m_as(UNITS["kt"])

    if not inplace:
        return ds
    else:
        return None


def compute_high_altitude(
    data_set: xr.Dataset, mask: t.Optional[xr.DataArray] = None, inplace: bool = False
) -> t.Optional[xr.Dataset]:
    """Compute U.S. Standard Atmosphere 1976 in high-altitude region.

    Parameters
    ----------
    data_set: Dataset
        Data set to compute.

    mask: DataArray, optional
        Mask to select the region of the data set to compute.
        By default, the mask selects the entire data set.

    inplace: bool, default False
        If ``True``, modifies ``data_set`` in place, else returns a copy of
        ``data_set``.

    Returns
    -------
    Dataset
        If ``inplace`` is True, returns nothing, else returns a copy of
        ``data_set``.
    """
    if mask is None:
        mask = xr.full_like(data_set.coords["z"], True, dtype=bool)

    if inplace:
        ds = data_set
    else:
        ds = data_set.copy(deep=True)

    altitudes = ds.coords["z"][mask]
    if len(altitudes) == 0:
        return ds

    z = ureg.Quantity(altitudes.values, "m")
    n = compute_number_densities_high_altitude(z)
    species = ["N2", "O", "O2", "Ar", "He", "H"]
    ni = np.array([n[s].m_as(1 / ureg.m ** 3) for s in species]) / ureg.m ** 3
    n_tot = np.sum(ni, axis=0)
    fi = ni / n_tot[np.newaxis, :]
    mi = (
        np.array([M[s].m_as(ureg.kg / ureg.mole) for s in species])
        * ureg.kg
        / ureg.mole
    )
    m = np.sum(fi * mi[:, np.newaxis], axis=0)
    t = compute_temperature_high_altitude(z)
    p = K * n_tot * t
    rho = np.sum(ni * mi[:, np.newaxis], axis=0) / NA
    g = compute_gravity(z)

    # assign data set with computed values
    ds["t"].loc[dict(z=altitudes)] = t.m_as(UNITS["t"])
    ds["p"].loc[dict(z=altitudes)] = p.m_as(UNITS["p"])
    ds["n_tot"].loc[dict(z=altitudes)] = n_tot.m_as(UNITS["n_tot"])

    for i, s in enumerate(SPECIES):
        if s in species:
            ds["n"][i].loc[dict(z=altitudes)] = n[s].m_as(UNITS["n"])

    ds["rho"].loc[dict(z=altitudes)] = rho.m_as(UNITS["rho"])
    ds["mv"].loc[dict(z=altitudes)] = (NA / n_tot).m_as(UNITS["mv"])
    ds["hp"].loc[dict(z=altitudes)] = (R * t / (g * m)).m_as(UNITS["hp"])
    ds["v"].loc[dict(z=altitudes)] = (np.sqrt(8.0 * R * t / (np.pi * m))).m_as(
        UNITS["v"]
    )
    ds["mfp"].loc[dict(z=altitudes)] = (
        np.sqrt(2.0) / (2.0 * np.pi * np.power(SIGMA, 2.0) * n_tot)
    ).m_as(UNITS["mfp"])
    ds["f"].loc[dict(z=altitudes)] = (
        4.0
        * NA
        * np.power(SIGMA, 2.0)
        * np.sqrt(np.pi * np.power(p, 2.0) / (R * m * t))
    ).m_as(UNITS["f"])

    if not inplace:
        return ds
    else:
        return None


def init_data_set(z: pint.Quantity) -> xr.Dataset:
    """Initialise data set.

    Parameters
    ----------
    z: quantity
        Altitudes.

    Returns
    -------
    Dataset
        Initialised data set.
    """
    data_vars = {}
    for var in VARIABLES:
        if var != "n":
            data_vars[var] = (
                DIMS[var],
                np.full(z.shape, np.nan),
                dict(units=UNITS[var], standard_name=STANDARD_NAME[var]),
            )
        else:
            data_vars[var] = (
                DIMS[var],
                np.full((len(SPECIES), len(z)), np.nan),
                dict(units=UNITS[var], standard_name=STANDARD_NAME["n"]),
            )

    coords = {
        "z": ("z", z.m_as(UNITS["z"]), dict(units=UNITS["z"])),
        "species": ("species", SPECIES),
    }

    # TODO: set function name in history field dynamically
    attrs = {
        "convention": "CF-1.8",
        "title": "U.S. Standard Atmosphere 1976",
        "history": (
            f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
            f" - data set creation - ussa1976.core.create"
        ),
        "source": f"ussa1976, version {__version__}",
        "references": (
            "U.S. Standard Atmosphere, 1976, NASA-TM-X-74335",
            "NOAA-S/T-76-1562",
        ),
    }

    return xr.Dataset(data_vars, coords, attrs)  # type: ignore


def compute_levels_temperature_and_pressure_low_altitude() -> t.Tuple[
    pint.Quantity, pint.Quantity
]:
    """Compute temperature and pressure at low-altitude region' levels.

    Returns
    -------
    tuple of quantity:
         Levels temperatures and pressures.
    """
    tb = [T0]
    pb = [P0]
    for i in range(1, 8):
        t_next = tb[i - 1] + LK[i - 1] * (H[i] - H[i - 1])
        tb.append(t_next)
        if LK[i - 1] == 0:
            p_next = compute_pressure_low_altitude_zero_gradient(
                h=H[i], hb=H[i - 1], pb=pb[i - 1], tb=tb[i - 1]
            )
        else:
            p_next = compute_pressure_low_altitude_non_zero_gradient(
                h=H[i], hb=H[i - 1], pb=pb[i - 1], tb=tb[i - 1], lkb=LK[i - 1]
            )
        pb.append(p_next)

    t_units = ureg.kelvin
    p_units = ureg.pascal
    t_values = np.array([t.m_as(t_units) for t in tb])
    p_values = np.array([p.m_as(p_units) for p in pb])
    return t_values * t_units, p_values * p_units


def compute_number_densities_high_altitude(
    altitudes: pint.Quantity,
) -> t.Dict[str, pint.Quantity]:
    """Compute number density of individual species in high-altitude region.

    Parameters
    ----------
    altitudes: quantity
        Altitudes.

    Returns
    -------
    dict
        Number densities of the individual species and total number density at
        the given altitudes.

    Notes
    -----
    A uniform altitude grid is generated and used for the computation of the
    integral as well as for the computation of the number densities of the
    individual species. This gridded data is then interpolated at the query
    ``altitudes`` using a linear interpolation scheme in logarithmic space.

    The number densities of the individual species are stored in a single
    2-D array where the first dimension is the gas species and the second
    dimension is the altitude. The species are in the following order:

    .. list-table:: Title
       :widths: 50 50
       :header-rows: 1

       * - Row
         - Species
       * - 0
         - N2
       * - 1
         - O
       * - 2
         - O2
       * - 3
         - Ar
       * - 4
         - He
       * - 5
         - H
    """
    # altitude grid
    grid = (
        np.concatenate(  # type: ignore
            (
                np.linspace(
                    start=Z7.m_as(ureg.km), stop=150.0, num=640, endpoint=False
                ),
                np.geomspace(
                    start=150.0, stop=Z12.m_as(ureg.km), num=100, endpoint=True
                ),
            )
        )
        * ureg.km
    )  # [km]

    # pre-computed variables
    m = compute_mean_molar_mass_high_altitude(z=grid)  # [kg/mol]
    g = compute_gravity(z=grid)  # [m / s^2]
    t = compute_temperature_high_altitude(grid)  # [K]
    dt_dz = compute_temperature_gradient_high_altitude(z=grid)  # [K/m]
    below_115 = grid.m_as(ureg.km) < 115.0
    k = eddy_diffusion_coefficient(grid[below_115])  # [m^2/s]

    n_grid = {}

    # molecular nitrogen
    y = m * g / (R * t)  # [m^-1]
    n_grid["N2"] = (
        N2_7
        * (T7 / t)
        * np.exp(
            -cumulative_trapezoid(y.m_as(1 / ureg.m), grid.m_as(ureg.m), initial=0.0)
        )
    )  # the factor 1000 is to convert km to m

    # atomic oxygen
    d = thermal_diffusion_coefficient(
        background=n_grid["N2"][below_115], temperature=t[below_115], a=A["O"], b=B["O"]
    )
    y = thermal_diffusion_term_atomic_oxygen(
        grid, g, t, dt_dz, d, k
    ) + velocity_term_atomic_oxygen(grid)
    n_grid["O"] = (
        O_7
        * (T7 / t)
        * np.exp(
            -cumulative_trapezoid(y.m_as(1 / ureg.m), grid.m_as(ureg.m), initial=0.0)
        )
    )

    # molecular oxygen
    d = thermal_diffusion_coefficient(
        background=n_grid["N2"][below_115],
        temperature=t[below_115],
        a=A["O2"],
        b=B["O2"],
    )
    y = thermal_diffusion_term("O2", grid, g, t, dt_dz, m, d, k) + velocity_term(
        "O2", grid
    )
    n_grid["O2"] = (
        O2_7
        * (T7 / t)
        * np.exp(
            -cumulative_trapezoid(y.m_as(1 / ureg.m), grid.m_as(ureg.m), initial=0.0)
        )
    )

    # argon
    background = (
        n_grid["N2"][below_115] + n_grid["O"][below_115] + n_grid["O2"][below_115]
    )
    d = thermal_diffusion_coefficient(
        background=background, temperature=t[below_115], a=A["Ar"], b=B["Ar"]
    )
    y = thermal_diffusion_term("Ar", grid, g, t, dt_dz, m, d, k) + velocity_term(
        "Ar", grid
    )
    n_grid["Ar"] = (
        AR_7
        * (T7 / t)
        * np.exp(
            -cumulative_trapezoid(y.m_as(1 / ureg.m), grid.m_as(ureg.m), initial=0.0)
        )
    )

    # helium
    background = (
        n_grid["N2"][below_115] + n_grid["O"][below_115] + n_grid["O2"][below_115]
    )
    d = thermal_diffusion_coefficient(
        background=background, temperature=t[below_115], a=A["He"], b=B["He"]
    )
    y = thermal_diffusion_term("He", grid, g, t, dt_dz, m, d, k) + velocity_term(
        "He", grid
    )
    n_grid["He"] = (
        HE_7
        * (T7 / t)
        * np.exp(
            -cumulative_trapezoid(y.m_as(1 / ureg.m), grid.m_as(ureg.m), initial=0.0)
        )
    )

    # hydrogen

    # below 500 km
    mask = (grid >= 150.0 * ureg.km) & (grid <= 500.0 * ureg.km)
    background = (
        n_grid["N2"][mask]
        + n_grid["O"][mask]
        + n_grid["O2"][mask]
        + n_grid["Ar"][mask]
        + n_grid["He"][mask]
    )
    d = thermal_diffusion_coefficient(background, t[mask], A["H"], B["H"])
    alpha = ALPHA["H"]
    _tau = tau_function(grid[mask], below_500=True)
    y = (PHI / d) * np.power(t[mask] / T11, 1 + alpha) * np.exp(_tau)
    print(y.units)
    integral_values = (
        cumulative_trapezoid(
            y[::-1].m_as(1 / ureg.m ** 4), grid[mask][::-1].m_as(ureg.m), initial=0.0
        )
        / ureg.m ** 3
    )
    integral_values = integral_values[::-1]
    n_below_500 = (
        (H_11 - integral_values) * np.power(T11 / t[mask], 1 + alpha) * np.exp(-_tau)
    )

    # above 500 km
    _tau = tau_function(grid[grid > 500.0 * ureg.km], below_500=False)
    n_above_500 = (
        H_11 * np.power(T11 / t[grid > 500.0 * ureg.km], 1 + alpha) * np.exp(-_tau)
    )

    n_grid["H"] = np.concatenate((n_below_500, n_above_500))  # type: ignore

    n = {
        s: log_interp1d(grid.m_as(ureg.m), n_grid[s].m_as(1 / ureg.m ** 3))(
            altitudes.m_as(ureg.m)
        )
        / ureg.m ** 3
        for s in ["N2", "O", "O2", "Ar", "He"]
    }

    # Below 150 km, the number density of atomic hydrogen is zero.
    n_h_below_150 = np.zeros(len(altitudes[altitudes < 150.0 * ureg.km])) / ureg.m ** 3
    n_h_above_150 = (
        log_interp1d(
            grid.m_as(ureg.km)[grid >= 150.0 * ureg.km],
            n_grid["H"].m_as(1 / ureg.m ** 3),
        )(altitudes.m_as(ureg.km)[altitudes >= 150.0 * ureg.km])
        / ureg.m ** 3
    )
    n["H"] = np.concatenate((n_h_below_150, n_h_above_150))  # type: ignore

    return n


def compute_mean_molar_mass_high_altitude(
    z: pint.Quantity,
) -> pint.Quantity:
    """Compute mean molar mass in high-altitude region.

    Parameters
    ----------
    z: quantity
        Altitude.

    Returns
    -------
    quantity
        Mean molar mass.
    """
    return np.where(z.m_as("km") <= 100.0, M0, M["N2"])


def compute_temperature_high_altitude(altitude: pint.Quantity) -> pint.Quantity:
    """Compute temperature in high-altitude region.

    Parameters
    ----------
    altitude: quantity
        Altitude.

    Returns
    -------
    quantity
        Temperature.
    """
    r0 = R0
    a = -76.3232  # K
    b = -19.9429  # km
    tc = 263.1905  # K

    def t(z: float) -> float:
        """Compute temperature at given altitude.

        Parameters
        ----------
        z: float
            Altitude [km].

        Returns
        -------
        float
            Temperature [K].

        Raises
        ------
        ValueError
            If the altitude is out of range.
        """
        if Z7.m_as(ureg.km) <= z <= Z8.m_as(ureg.km):
            return T7.m_as(ureg.K)  # type: ignore
        elif Z8.m_as(ureg.km) < z <= Z9.m_as(ureg.km):
            return tc + a * float(
                np.sqrt(1.0 - np.power((z - Z8.m_as(ureg.km)) / b, 2.0))
            )
        elif Z9.m_as(ureg.km) < z <= Z10.m_as(ureg.km):
            t9 = T9.m_as(ureg.K)
            lk9 = LK9.m_as(ureg.K / ureg.km)
            return t9 + lk9 * (z - Z9.m_as(ureg.km))  # type: ignore
        elif Z10.m_as(ureg.km) < z <= Z12.m_as(ureg.km):
            t_inf = TINF.m_as(ureg.K)
            t10 = T10.m_as(ureg.K)
            return t_inf - (t_inf - t10) * float(  # type: ignore
                np.exp(
                    -LAMBDA.m_as(1 / ureg.km)
                    * (z - Z10.m_as(ureg.km))
                    * (r0.m_as(ureg.km) + Z10.m_as(ureg.km))
                    / (r0.m_as(ureg.km) + z)
                )
            )
        else:
            raise ValueError("altitude value is out of range")

    return np.array(np.vectorize(t)(altitude.m_as(ureg.km))) * ureg.K


def compute_temperature_gradient_high_altitude(z: pint.Quantity) -> pint.Quantity:
    """Compute temperature gradient in high-altitude region.

    Parameters
    ----------
    z: quantity
        Altitude.

    Returns
    -------
    quantity
        Temperature gradient.
    """
    a = -76.3232  # [dimensionless]
    b = -19.9429  # km

    def gradient(z_value: float) -> float:
        """Compute temperature gradient at given altitude.

        Parameters
        ----------
        z_value: float
            Altitude [km].

        Raises
        ------
        ValueError
            When altitude is out of bounds.

        Returns
        -------
        float
            Temperature gradient [K/km].
        """
        if Z7.m_as("km") <= z_value <= Z8.m_as("km"):
            return float(LK7.m_as("K/km"))  # type: ignore
        elif Z8.m_as("km") < z_value <= Z9.m_as("km"):
            return float(  # type: ignore
                -a
                / b
                * ((z_value - Z8.m_as("km")) / b)
                / float(np.sqrt(1 - np.square((z_value - Z8.m_as("km")) / b)))
            )
        elif Z9.m_as("km") < z_value <= Z10.m_as("km"):
            return float(LK9.m_as("K/km"))  # type: ignore
        elif Z10.m_as("km") < z_value <= Z12.m_as("km"):
            zeta = (
                (z_value - Z10.m_as("km"))
                * (R0.m_as("km") + Z10.m_as("km"))
                / (R0.m_as("km") + z_value)
            )  # [km]
            return float(  # type: ignore
                LAMBDA.m_as("km^-1")
                * (TINF - T10).m_as("K")
                * float(
                    np.square(
                        (R0.m_as("km") + Z10.m_as("km")) / (R0.m_as("km") + z_value)
                    )
                )
                * float(np.exp(-LAMBDA.m_as("km^-1") * zeta))
            )

        else:
            raise ValueError(
                f"altitude z ({z_value}) out of range, should be in ["
                f"{Z7.m_as('km')}, {Z12.m_as('km')}]"
            )

    z_values = np.array(z.m_as("km"), dtype=float)
    return np.array(np.vectorize(gradient)(z_values)) * ureg.K / ureg.km


def thermal_diffusion_coefficient(
    background: pint.Quantity,
    temperature: pint.Quantity,
    a: pint.Quantity,
    b: pint.Quantity,
) -> pint.Quantity:
    r"""Compute thermal diffusion coefficient values in high-altitude region.

    Parameters
    ----------
    background: quantity
        Background number density.

    temperature: quantity
        Temperature.

    a: quantity
        Thermal diffusion constant :math:`a`.

    b: quantity
        Thermal diffusion constant :math:`b`.

    Returns
    -------
    quantity
        Thermal diffusion coefficient.
    """
    return (a / background) * np.power(
        temperature.m_as(ureg.K) / 273.15, b.m_as(ureg.dimensionless)
    )


def eddy_diffusion_coefficient(z: pint.Quantity) -> pint.Quantity:
    r"""Compute Eddy diffusion coefficient in high-altitude region.

    Parameters
    ----------
    z: quantity
        Altitude.

    Returns
    -------
    quantity
        Eddy diffusion coefficient.

    Notes
    -----
    Valid in the altitude region :math:`86 \leq z \leq 150` km.
    """
    return np.where(
        z.m_as(ureg.km) < 95.0,
        K_7,
        K_7 * np.exp(1.0 - (400.0 / (400.0 - np.square(z.m_as(ureg.km) - 95.0)))),
    )


def f_below_115_km(
    g: pint.Quantity,
    t: pint.Quantity,
    dt_dz: pint.Quantity,
    m: pint.Quantity,
    mi: pint.Quantity,
    alpha: pint.Quantity,
    d: pint.Quantity,
    k: pint.Quantity,
) -> pint.Quantity:
    r"""Evaluate function :math:`f` below 115 km altitude.

    Evaluates the function :math:`f` defined by equation (36) in
    :cite:`NASA1976USStandardAtmosphere` in the altitude region :math:`86
    \leq z \leq 115` km.

    Parameters
    ----------
    g: quantity
        Gravity values at the different altitudes.

    t: quantity
        Temperature values at the different altitudes.

    dt_dz: quantity
        Temperature gradient values at the different altitudes.

    m: quantity
        Molar mass.

    mi: quantity
        Species molar masses.

    alpha: quantity
        Alpha thermal diffusion constant.

    d: quantity
        Thermal diffusion coefficient values at the different altitudes.

    k: quantity
        Eddy diffusion coefficient values at the different altitudes.

    Returns
    -------
    quantity
        Function :math:`f` at the different altitudes.
    """
    term_1 = g * d / ((d + k) * (R * t))
    term_2 = mi + (m * k) / d + (alpha * R * dt_dz) / g
    return term_1 * term_2


def f_above_115_km(
    g: pint.Quantity,
    t: pint.Quantity,
    dt_dz: pint.Quantity,
    mi: pint.Quantity,
    alpha: pint.Quantity,
) -> pint.Quantity:
    r"""Evaluate function :math:`f` above 115 km altitude.

    Evaluate the function :math:`f` defined by equation (36) in
    :cite:`NASA1976USStandardAtmosphere` in the altitude region :math:`115 \lt
    z \leq 1000` km.

    Parameters
    ----------
    g: quantity
        Gravity at the different altitudes.

    t: quantity
        Temperature at the different altitudes.

    dt_dz: quantity
        Temperature gradient at the different altitudes.

    mi: quantity
        Species molar masses.

    alpha: quantity
        Alpha thermal diffusion constant.

    Returns
    -------
    quantity
        Function :math:`f` at the different altitudes.
    """
    return (g / (R * t)) * (mi + ((alpha * R) / g) * dt_dz)


def thermal_diffusion_term(
    species: str,
    grid: pint.Quantity,
    g: pint.Quantity,
    t: pint.Quantity,
    dt_dz: pint.Quantity,
    m: pint.Quantity,
    d: pint.Quantity,
    k: pint.Quantity,
) -> pint.Quantity:
    """Compute thermal diffusion term of given species in high-altitude region.

    Parameters
    ----------
    species: str
        Species.

    grid: quantity
        Altitude grid.

    g: quantity
        Gravity values on the altitude grid.

    t: quantity
        Temperature values on the altitude grid.

    dt_dz: quantity
        Temperature gradient values on the altitude grid.

    m: quantity
        Values of the mean molar mass on the altitude grid.

    d: quantity
        Molecular diffusion coefficient values on the altitude grid,
        for altitudes strictly less than 115 km.

    k: quantity
        Eddy diffusion coefficient values on the altitude grid, for
        altitudes strictly less than 115 km.

    Returns
    -------
    quantity
        Thermal diffusion term.
    """
    below_115_km = grid < 115.0 * ureg.km
    fo1 = f_below_115_km(
        g[below_115_km],
        t[below_115_km],
        dt_dz[below_115_km],
        m[below_115_km],
        M[species],
        ALPHA[species],
        d,
        k,
    )
    above_115_km = grid >= 115.0 * ureg.km
    fo2 = f_above_115_km(
        g[above_115_km],
        t[above_115_km],
        dt_dz[above_115_km],
        M[species],
        ALPHA[species],
    )
    return np.concatenate((fo1, fo2))  # type: ignore


def thermal_diffusion_term_atomic_oxygen(
    grid: pint.Quantity,
    g: pint.Quantity,
    t: pint.Quantity,
    dt_dz: pint.Quantity,
    d: pint.Quantity,
    k: pint.Quantity,
) -> pint.Quantity:
    """Compute oxygen thermal diffusion term in high-altitude region.

    Parameters
    ----------
    grid: quantity
        Altitude grid.

    g: quantity
        Gravity values on the altitude grid.

    t: quantity
        Temperature values on the altitude grid.

    dt_dz: quantity
        Temperature values gradient on the altitude grid.

    d: quantity
        Thermal diffusion coefficient on the altitude grid.

    k: quantity
        Eddy diffusion coefficient values on the altitude grid.

    Returns
    -------
    quantity
        Thermal diffusion term.
    """
    mask1, mask2 = grid < 115.0 * ureg.km, grid >= 115.0 * ureg.km
    x1 = f_below_115_km(
        g=g[mask1],
        t=t[mask1],
        dt_dz=dt_dz[mask1],
        m=M["N2"],
        mi=M["O"],
        alpha=ALPHA["O"],
        d=d,
        k=k,
    )
    x2 = f_above_115_km(
        g=g[mask2], t=t[mask2], dt_dz=dt_dz[mask2], mi=M["O"], alpha=ALPHA["O"]
    )
    return np.concatenate((x1, x2))  # type: ignore


def velocity_term_hump(
    z: pint.Quantity,
    q1: pint.Quantity,
    q2: pint.Quantity,
    u1: pint.Quantity,
    u2: pint.Quantity,
    w1: pint.Quantity,
    w2: pint.Quantity,
) -> pint.Quantity:
    r"""Compute transport term.

    Compute the transport term given by equation (37) in
    :cite:`NASA1976USStandardAtmosphere`.

    Parameters
    ----------
    z: quantity
        Altitude.

    q1: quantity
        Q constant.

    q2: quantity
        q constant.

    u1: quantity
        U constant.

    u2: quantity
        u constant.

    w1: quantity
        W constant.

    w2: quantity
        w constant.

    Returns
    -------
    quantity:
        Transport term.

    Notes
    -----
    Valid in the altitude region: 86 km :math:`\leq z \leq` 150 km.
    """
    return q1 * np.square(z - u1) * np.exp(
        -w1.m_as(1 / ureg.km ** 3) * np.power((z - u1).m_as(ureg.km), 3.0)
    ) + q2 * np.square(u2 - z) * np.exp(
        -w2.m_as(1 / ureg.km ** 3) * np.power((u2 - z).m_as(ureg.km), 3.0)
    )


def velocity_term_no_hump(
    z: pint.Quantity, q1: pint.Quantity, u1: pint.Quantity, w1: pint.Quantity
) -> pint.Quantity:
    r"""Compute transport term.

    Compute the transport term given by equation (37) in
    :cite:`NASA1976USStandardAtmosphere` where the second term is zero.

    Parameters
    ----------
    z: quantity
        Altitude.

    q1: quantity
        Q constant.

    u1: quantity
        U constant.

    w1: quantity
        W constant.

    Returns
    -------
    quantity
        Transport term.

    Notes
    -----
    Valid in the altitude region :math:`86 \leq z \leq 150` km.
    """
    return (
        q1
        * np.square(z - u1)
        * np.exp(-w1.m_as(1 / ureg.km ** 3) * np.power((z - u1).m_as(ureg.km), 3.0))
    )


def velocity_term(species: str, grid: pint.Quantity) -> pint.Quantity:
    """Compute velocity term of a given species in high-altitude region.

    Parameters
    ----------
    species: str
        Species.

    grid: quantity
        Altitude grid.

    Returns
    -------
    quantity
        Velocity term.

    Notes
    -----
    Not valid for atomic oxygen. See :func:`velocity_term_atomic_oxygen`
    """
    x1 = velocity_term_no_hump(
        z=grid[grid <= 150.0 * ureg.km], q1=Q1[species], u1=U1[species], w1=W1[species]
    )

    # Above 150 km, the velocity term is neglected, as indicated at p. 14 in
    # :cite:`NASA1976USStandardAtmosphere`
    x2 = np.zeros(len(grid[grid > 150.0 * ureg.km]))
    return np.concatenate((x1, x2))  # type: ignore


def velocity_term_atomic_oxygen(grid: pint.Quantity) -> pint.Quantity:
    """Compute velocity term of atomic oxygen in high-altitude region.

    Parameters
    ----------
    grid: quantity
        Altitude grid.

    Returns
    -------
    quantity
        Velocity term.
    """
    mask1, mask2 = grid <= 150.0 * ureg.km, grid > 150.0 * ureg.km
    x1 = np.where(
        grid[mask1] <= 97.0 * ureg.km,
        velocity_term_hump(
            z=grid[mask1],
            q1=Q1["O"],
            q2=Q2["O"],
            u1=U1["O"],
            u2=U2["O"],
            w1=W1["O"],
            w2=W2["O"],
        ),
        velocity_term_no_hump(z=grid[mask1], q1=Q1["O"], u1=U1["O"], w1=W1["O"]),
    )

    x2 = np.zeros(len(grid[mask2]))
    return np.concatenate((x1, x2))  # type: ignore


def tau_function(
    z_grid: pint.Quantity, below_500: bool = True
) -> npt.NDArray[np.float64]:
    r"""Compute :math:`\tau` function.

    Compute integral given by equation (40) in
    :cite:`NASA1976USStandardAtmosphere` at each point of an altitude grid.

    Parameters
    ----------
    z_grid: quantity
        Altitude grid (values sorted by ascending order) to use for integration.

    below_500: bool, default True
        ``True`` if altitudes in ``z_grid`` are lower than 500 km, False
        otherwise.

    Returns
    -------
    ndarray
        Integral evaluations [dimensionless].

    Notes
    -----
    Valid for 150 km :math:`leq z \leq` 500 km.
    """
    if below_500:
        z_grid = z_grid[::-1]

    y = (
        M["H"]
        * compute_gravity(z=z_grid)
        / (R * compute_temperature_high_altitude(altitude=z_grid))
    )
    integral_values = cumulative_trapezoid(
        y.m_as(1 / ureg.m), z_grid.m_as(ureg.m), initial=0.0
    )

    if below_500:
        return integral_values[::-1]  # type: ignore
    else:
        return integral_values  # type: ignore


def log_interp1d(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> t.Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Compute linear interpolation of :math:`y(x)` in logarithmic space.

    Parameters
    ----------
    x: ndarray
        1-D array of real values.

    y: ndarray
        N-D array of real values. The length of y along the interpolation axis
        must be equal to the length of x.

    Returns
    -------
    callable
        Interpolating function.
    """
    logx = np.log10(x)
    logy = np.log10(y)
    lin_interp = interp1d(logx, logy, kind="linear")

    def log_interp(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array(np.power(10.0, lin_interp(np.log10(z))))

    return log_interp


def compute_pressure_low_altitude(
    h: pint.Quantity,
    pb: pint.Quantity,
    tb: pint.Quantity,
) -> pint.Quantity:
    """Compute pressure in low-altitude region.

    Parameters
    ----------
    h: quantity
        Geopotential height.

    pb: quantity
        Levels pressure.

    tb: quantity
        Levels temperature.

    Returns
    -------
    quantity
        Pressure.
    """
    # we create a mask for each layer
    h_units = ureg.m
    masks = [
        ma.masked_inside(  # type: ignore
            h.m_as(h_units), H[i - 1].m_as(h_units), H[i].m_as(h_units)
        ).mask
        for i in range(1, 8)
    ]

    # for each layer, we evaluate the pressure based on whether the
    # temperature gradient is zero or non-zero
    p = np.empty(len(h))
    p_units = ureg.pascal
    for i, mask in enumerate(masks):
        if LK[i].magnitude == 0:
            p[mask] = compute_pressure_low_altitude_zero_gradient(
                h=h[mask], hb=H[i], pb=pb[i], tb=tb[i]
            ).m_as(p_units)
        else:
            p[mask] = compute_pressure_low_altitude_non_zero_gradient(
                h=h[mask], hb=H[i], pb=pb[i], tb=tb[i], lkb=LK[i]
            ).m_as(p_units)
    return p * p_units


def compute_pressure_low_altitude_zero_gradient(
    h: pint.Quantity,
    hb: pint.Quantity,
    pb: pint.Quantity,
    tb: pint.Quantity,
) -> pint.Quantity:
    """Compute pressure in low-altitude zero temperature gradient region.

    Parameters
    ----------
    h: quantity
        Geopotential height.

    hb: quantity
        Geopotential height at the bottom of the layer.

    pb: quantity
        Pressure at the bottom of the layer.

    tb: quantity
        Temperature at the bottom of the layer.

    Returns
    -------
    quantity
        Pressure.
    """
    return pb * np.exp(-G0 * M0 * (h - hb) / (R * tb))


def compute_pressure_low_altitude_non_zero_gradient(
    h: pint.Quantity,
    hb: pint.Quantity,
    pb: pint.Quantity,
    tb: pint.Quantity,
    lkb: pint.Quantity,
) -> pint.Quantity:
    """Compute pressure in low-altitude non-zero temperature gradient region.

    Parameters
    ----------
    h: quantity
        Geopotential height.

    hb: quantity
        Geopotential height at the bottom of the layer.

    pb: quantity
        Pressure at the bottom of the layer.

    tb: quantity
        Temperature at the bottom of the layer.

    lkb: quantity
        Temperature gradient in the layer.

    Returns
    -------
    quantity
        Pressure.
    """
    return pb * np.power(tb / (tb + lkb * (h - hb)), G0 * M0 / (R * lkb))


def compute_temperature_low_altitude(
    h: pint.Quantity,
    tb: pint.Quantity,
) -> pint.Quantity:
    """Compute temperature in low-altitude region.

    Parameters
    ----------
    h: quantity
        Geopotential height.

    tb: quantity
        Levels temperature.

    Returns
    -------
    quantity
        Temperature.
    """
    # we create a mask for each layer
    h_units = ureg.m
    masks = [
        ma.masked_inside(  # type: ignore
            h.m_as(h_units), H[i - 1].m_as(h_units), H[i].m_as(h_units)
        ).mask
        for i in range(1, 8)
    ]

    # for each layer, we evaluate the pressure based on whether the
    # temperature gradient is zero or not
    t = np.empty(len(h))
    t_units = ureg.kelvin
    for i, mask in enumerate(masks):
        if LK[i].magnitude == 0:
            t[mask] = tb[i].m_as(t_units)
        else:
            t[mask] = (tb[i] + LK[i] * (h[mask] - H[i])).m_as(t_units)
    return t * t_units


def to_altitude(h: pint.Quantity) -> pint.Quantity:
    """Convert geopotential height to (geometric) altitude.

    Parameters
    ----------
    h: quantity
        Geopotential altitude.

    Returns
    -------
    quantity
        Altitude.
    """
    return R0 * h / (R0 - h)


def to_geopotential_height(z: pint.Quantity) -> pint.Quantity:
    """Convert altitude to geopotential height.

    Parameters
    ----------
    z: quantity
        Altitude.

    Returns
    -------
    quantity
        Geopotential height.
    """
    return R0 * z / (R0 + z)


def compute_gravity(z: pint.Quantity) -> pint.Quantity:
    """Compute gravity.

    Parameters
    ----------
    z : quantity
        Altitude.

    Returns
    -------
    quantity
        Gravity.
    """
    return G0 * np.power((R0 / (R0 + z)), 2.0)
