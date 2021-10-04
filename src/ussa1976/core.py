"""
US Standard Atmosphere 1976 thermophysical model, according to
:cite:`NASA1976USStandardAtmosphere`.
"""
import typing as t
from datetime import datetime

import numpy as np
import numpy.ma as ma
import numpy.typing as npt
import pint
import xarray as xr
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

from ussa1976 import __version__, ureg

# ------------------------------------------------------------------------------
#
# Atmospheric vertical profile data set generator
#
# ------------------------------------------------------------------------------


@ureg.wraps(ret=None, args="m", strict=False)
def make(
    levels: t.Union[ureg.Quantity, npt.NDArray[np.float64]] = ureg.Quantity(
        np.linspace(0.0, 1e5, 51), "m"
    )
) -> xr.Dataset:
    """Make U.S. Standard Atmosphere 1976.

    .. note::
        The pressure, temperature and number densities given in each layer of
        the altitude mesh are computed at the altitude of the layers centers.
        In other words, the layer's middle is taken as the altitude
        representative of the whole layer. For example, in a layer with lower
        and upper altitudes of 1000 and 2000 m, the thermophysical variables
        are computed at the altitude of 1500 m.

    Parameters
    ----------
    levels: :class:`~numpy.ndarray`
        Levels altitudes [m].
        The values must be sorted by increasing order.
        Valid range: 0 to 1e6 m.

    Returns
    -------
    :class:`~xarray.Dataset`
        Data set holding the values of the pressure, temperature,
        total number density and number densities of the individual
        gas species in each layer.
    """

    if np.any(levels > 1e6) or np.any(levels < 0.0):
        raise ValueError("Levels altitudes must be in [0, 1e6] m.")

    z_layer = (levels[:-1] + levels[1:]) / 2

    # create the US76 data set
    ds = create(
        ureg.Quantity(z_layer, "m"),
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
        z_layer,
        dict(
            standard_name="layer_altitude",
            long_name="layer altitude",
            units="m",
        ),
    )
    thermoprops_ds.coords["z_level"] = (
        "z_level",
        levels,
        dict(
            standard_name="level_altitude",
            long_name="level altitude",
            units="m",
        ),
    )
    thermoprops_ds.attrs = dict(
        convention="CF-1.8",
        title="U.S. Standard Atmosphere 1976",
        history=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
        f"data creation - eradiate.scenes.atmosphere.us76.make_profile",
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
K = 1.380622e-23  # [J/K]

# Molar masses of the individual species
M = {
    "N2": 0.0280134,
    "O2": 0.0319988,
    "Ar": 0.039948,
    "CO2": 0.04400995,
    "Ne": 0.020183,
    "He": 0.0040026,
    "Kr": 0.08380,
    "Xe": 0.13130,
    "CH4": 0.01604303,
    "H2": 0.00201594,
    "O": 0.01599939,
    "H": 0.00100797,
}  # [kg/mol]

# Sea level mean (mixture) molar mass
M0 = 0.028964425278793997  # [kg/mol]

# Avogadro number
NA = 6.022169e23  # [mol^-1]

# Universal gas constant
R = 8.31432  # [J/(mol*K)]

# Sea level volume fractions of the gas species present below 86 km
F = {
    "N2": 0.78084,
    "O2": 0.209476,
    "Ar": 0.00934,
    "CO2": 0.000314,
    "Ne": 0.00001818,
    "He": 0.00000524,
    "Kr": 0.00000114,
    "Xe": 0.000000087,
    "CH4": 0.000002,
    "H2": 0.0000005,
}  # [dimensionless]

# Sea level gravity
G0 = 9.80665  # [m/s^2]

# Geopotential altitudes of the layers' boundaries (below 86 km)
H = [
    0.0,
    11e3,
    20e3,
    32e3,
    47e3,
    51e3,
    71e3,
    84852.05,
]  # [m]

# Temperature gradients in the seven layers (below 86 km)
LK = [
    -0.0065,
    0.0,
    0.0010,
    0.0028,
    0.0,
    -0.0028,
    -0.0020,
]  # [K/m]

# Pressure at sea level
P0 = 101325.0  # [Pa]

# Effective Earth radius
R0 = 6.356766e6  # [m]

# Temperature at sea level
T0 = 288.15  # [K]
S = 110.4  # [K]
BETA = 1.458e6  # [kg/(m*s*K^1/2)]
GAMMA = 1.40  # [dimensionless]
SIGMA = 3.65e-10  # [m]

# Thermal diffusion constants of the individual species present above 86 km
ALPHA = {
    "N2": 0.0,  # [dimensionless]
    "O": 0.0,  # [dimensionless]
    "O2": 0.0,  # [dimensionless]
    "Ar": 0.0,  # [dimensionless]
    "He": -0.4,  # [dimensionless]
    "H": -0.25,  # [dimensionless]
}
A = {
    "N2": None,
    "O": 6.986e20,  # [m^-1*s^-1]
    "O2": 4.863e20,  # [m^-1*s^-1]
    "Ar": 4.487e20,  # [m^-1*s^-1]
    "He": 1.7e21,  # [m^-1*s^-1]
    "H": 3.305e21,  # [m^-1*s^-1]
}
B = {
    "N2": None,
    "O": 0.75,  # [dimensionless]
    "O2": 0.75,  # [dimensionless]
    "Ar": 0.87,  # [dimensionless]
    "He": 0.691,  # [dimensionless]
    "H": 0.5,  # [dimensionless]
}

# Eddy diffusion coefficients
K_7 = 1.2e2  # [m^2/s]
K_10 = 0.0  # [m^2/s]

# Vertical transport constants of the individual species present above 86 km
Q1 = {
    "O": -5.809644e-4,  # [km^-3]
    "O2": 1.366212e-4,  # [km^-3]
    "Ar": 9.434079e-5,  # [km^-3]
    "He": -2.457369e-4,  # [km^-3]
}
Q2 = {
    "O": -3.416248e-3,  # [km^-3], /!\ above 97 km, Q2 = 0.
    "O2": 0.0,  # [km^-3]
    "Ar": 0.0,  # [km^-3]
    "He": 0.0,  # [km^-3]
}
U1 = {
    "O": 56.90311,  # [km]
    "O2": 86.0,  # [km]
    "Ar": 86.0,  # [km]
    "He": 86.0,  # [km]
}
U2 = {"O": 97.0, "O2": None, "Ar": None, "He": None}  # [km]
W1 = {
    "O": 2.706240e-5,  # [km^-3]
    "O2": 8.333333e-5,  # [km^-3]
    "Ar": 8.333333e-5,  # [km^-3]
    "He": 6.666667e-4,  # [km^-3]
}
W2 = {"O": 5.008765e-4, "O2": None, "Ar": None, "He": None}  # [km^-3]

# Altitudes of the levels delimiting 5 layers above 86 km
Z7 = 86.0  # [km]
Z8 = 91.0  # [km]
Z9 = 110.0  # [km]
Z10 = 120.0  # [km]
Z11 = 500.0  # [km]
Z12 = 1000.0  # [km]

# Temperature at the different levels above 86 km
T7 = 186.8673  # [K]
T9 = 240.0  # [K]
T10 = 360.0  # [K]
T11 = 999.2356  # [K]
TINF = 1000.0  # [K]
LAMBDA = 0.01875  # [km^-1]

# Temperature gradients
LK7 = 0.0  # [K/km]
LK9 = 12.0  # [K/km]

# Molecular nitrogen at altitude = Z7
N2_7 = 1.129794e20  # [m^-3]

# Atomic oxygen at altitude = Z7
O_7 = 8.6e16  # [m^-3]

# Molecular oxygen at altitude = Z7
O2_7 = 3.030898e19  # [m^-3]

# Argon at altitude = Z7
AR_7 = 1.351400e18  # [m^-3]

# Helium at altitude = Z7 (assumes typo at page 13)
HE_7 = 7.5817e14  # [m^-3]

# Hydrogen at altitude = Z7
H_11 = 8.0e10  # [m^-3]

# Vertical flux
PHI = 7.2e11  # [m^-2 * s^-1]

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
    "mv": "m^3",
    "hp": "m",
    "v": "m/s",
    "mfp": "m",
    "f": "s^-1",
    "cs": "m/s",
    "mu": "kg/(m*s)",
    "nu": "m^2/s",
    "kt": "J/m",
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


@ureg.wraps(ret=None, args=("m", None), strict=False)
def create(
    z: t.Union[ureg.Quantity, npt.NDArray[np.float64]],
    variables: t.Optional[t.List[str]] = None,
) -> xr.Dataset:
    """Creates a US Standard Atmosphere 1976 data set.

    .. warning::
        The returned U.S. Standard Atmosphere 1976 data set is not an
        atmospheric vertical profile data set. See
        :func:`eradiate.thermoprops.us76.make_profile`
        if you are interested in generating an atmospheric vertical profile
        based on the U.S. Standard Atmosphere 1976 model.

    Parameters
    ----------
    z: :class:`~numpy.ndarray`
        1-D array with altitude values [m].

    variable: list, optional
        Names of the variables to compute.

    Returns
    -------
    class:`~xarray.Dataset`:
        Data set holding the values of the different atmospheric variables.
    """

    if np.any(z < 0.0):
        raise ValueError("altitude values must be greater than or equal to " "zero")

    if np.any(z > 1000000.0):
        raise ValueError("altitude values must be less then or equal to 1e6 m")

    if variables is None:
        variables = VARIABLES
    else:
        for var in variables:
            if var not in VARIABLES:
                raise ValueError(var, " is not a valid variable name")

    # initialise data set
    ds = init_data_set(ureg.Quantity(z, "m"))

    # compute the model in the low-altitude region
    compute_low_altitude(ds, ds.coords["z"] <= 86000.0, inplace=True)

    # compute the model in the high-altitude region
    compute_high_altitude(ds, ds.coords["z"] > 86000.0, inplace=True)

    # replace all np.nan with 0. in number densities values
    n = ds.n.values
    n[np.isnan(n)] = 0.0
    ds.n.values = n

    # list names of variables to drop from the data set
    names = []
    for var in ds.data_vars:
        if var not in variables:
            names.append(var)

    return ds.drop_vars(names)  # type: ignore


def compute_low_altitude(
    data_set: xr.Dataset, mask: t.Optional[xr.DataArray] = None, inplace: bool = False
) -> t.Optional[xr.Dataset]:
    """Computes the US Standard Atmosphere 1976 in the low-altitude region.

    Parameters
    ----------
    data_set: :class:`~xarray.Dataset`
        Data set to compute.

    mask: :class:`~xarray.DataArray`
        Mask to select the region of the data set to compute.
        By default, the mask selects the entire data set.

    inplace: bool
        If true, modifies ``data_set`` in place, else returns a copy of
        ``data_set``.

    Returns
    -------
    :class:`~xarray.Dataset`
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
    z = altitudes.values

    # compute levels temperature and pressure values
    tb, pb = compute_levels_temperature_and_pressure_low_altitude()

    # compute geopotential height, temperature and pressure
    h = to_geopotential_height(z)
    t = compute_temperature_low_altitude(h, tb)
    p = compute_pressure_low_altitude(h, pb, tb)

    # compute the auxiliary atmospheric variables
    n_tot = NA * p / (R * t)
    rho = p * M0 / (R * t)
    g = compute_gravity(z)
    mu = BETA * np.power(t, 1.5) / (t + S)

    # assign data set with computed values
    ds["t"].loc[dict(z=altitudes)] = t
    ds["p"].loc[dict(z=altitudes)] = p
    ds["n_tot"].loc[dict(z=altitudes)] = n_tot

    species = ["N2", "O2", "Ar", "CO2", "Ne", "He", "Kr", "Xe", "CH4", "H2"]
    for i, s in enumerate(SPECIES):
        if s in species:
            ds["n"][i].loc[dict(z=altitudes)] = F[s] * n_tot

    ds["rho"].loc[dict(z=altitudes)] = rho
    ds["mv"].loc[dict(z=altitudes)] = NA / n_tot
    ds["hp"].loc[dict(z=altitudes)] = R * t / (g * M0)
    ds["v"].loc[dict(z=altitudes)] = np.sqrt(8.0 * R * t / (np.pi * M0))
    ds["mfp"].loc[dict(z=altitudes)] = np.sqrt(2.0) / (
        2.0 * np.pi * np.power(SIGMA, 2.0) * n_tot
    )
    ds["f"].loc[dict(z=altitudes)] = (
        4.0
        * NA
        * np.power(SIGMA, 2.0)
        * np.sqrt(np.pi * np.power(p, 2.0) / (R * M0 * t))
    )
    ds["cs"].loc[dict(z=altitudes)] = np.sqrt(GAMMA * R * t / M0)
    ds["mu"].loc[dict(z=altitudes)] = mu
    ds["nu"].loc[dict(z=altitudes)] = mu / rho
    ds["kt"].loc[dict(z=altitudes)] = (
        2.64638e-3 * np.power(t, 1.5) / (t + 245.4 * np.power(10.0, -12.0 / t))
    )

    if not inplace:
        return ds
    else:
        return None


def compute_high_altitude(
    data_set: xr.Dataset, mask: t.Optional[xr.DataArray] = None, inplace: bool = False
) -> t.Optional[xr.Dataset]:
    """Computes the US Standard Atmosphere 1976 in the high-altitude region.

    Parameters
    ----------
    data_set: :class:`~xarray.Dataset`
        Data set to compute.

    mask: :class:`~xarray.DataArray`:
        Mask to select the region of the data set to compute.
        By default, the mask selects the entire data set.

    inplace: bool
        If true, modifies ``data_set`` in place, else returns a copy of
        ``data_set``.
        Default: False.

    Returns
    -------
    :class:`~xarray.Dataset`
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
    ni = np.array([n[s] for s in species])
    n_tot = np.sum(ni, axis=0)
    fi = ni / n_tot[np.newaxis, :]
    mi = np.array([M[s] for s in species])
    m = np.sum(fi * mi[:, np.newaxis], axis=0)
    t = compute_temperature_high_altitude(z)
    p = K * n_tot * t
    rho = np.sum(ni * mi[:, np.newaxis], axis=0) / NA
    g = compute_gravity(z)

    # assign data set with computed values
    ds["t"].loc[dict(z=altitudes)] = t
    ds["p"].loc[dict(z=altitudes)] = p
    ds["n_tot"].loc[dict(z=altitudes)] = n_tot

    for i, s in enumerate(SPECIES):
        if s in species:
            ds["n"][i].loc[dict(z=altitudes)] = n[s]

    ds["rho"].loc[dict(z=altitudes)] = rho
    ds["mv"].loc[dict(z=altitudes)] = NA / n_tot
    ds["hp"].loc[dict(z=altitudes)] = R * t / (g * m)
    ds["v"].loc[dict(z=altitudes)] = np.sqrt(8.0 * R * t / (np.pi * m))
    ds["mfp"].loc[dict(z=altitudes)] = np.sqrt(2.0) / (
        2.0 * np.pi * np.power(SIGMA, 2.0) * n_tot
    )
    ds["f"].loc[dict(z=altitudes)] = (
        4.0
        * NA
        * np.power(SIGMA, 2.0)
        * np.sqrt(np.pi * np.power(p, 2.0) / (R * m * t))
    )

    if not inplace:
        return ds
    else:
        return None


@ureg.wraps(ret=None, args="m", strict=False)
def init_data_set(z: t.Union[ureg.Quantity, npt.NDArray[np.float64]]) -> xr.Dataset:
    """Initialises the data set.

    Parameters
    ----------
    z: :class:`~ureg.Quantity`
        Altitudes values [m]

    Returns
    -------
    :class:`~xarray.Dataset`:
        Initialised data set.
    """
    data_vars = {}
    for var in VARIABLES:
        if var != "n":
            try:
                data_vars[var] = (
                    DIMS[var],
                    np.full(z.shape, np.nan),
                    {"units": UNITS[var], "standard_name": STANDARD_NAME[var]},
                )
            except KeyError:
                data_vars[var] = (
                    DIMS[var],
                    np.full(z.shape, np.nan),
                    {"units": UNITS[var], "standard_name": STANDARD_NAME[var]},
                )
        else:
            data_vars[var] = (
                DIMS[var],
                np.full((len(SPECIES), len(z)), np.nan),
                {"units": UNITS[var], "standard_name": STANDARD_NAME["n"]},
            )

    coords = {"z": ("z", z, {"units": UNITS["z"]}), "species": ("species", SPECIES)}

    # TODO: set function name in history field dynamically
    attrs = {
        "convention": "CF-1.8",
        "title": "U.S. Standard Atmosphere 1976",
        "history": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - data creation - "
        f"eradiate.scenes.atmosphere.us76.create",
        "source": f"eradiate, version {__version__}",
        "references": "U.S. Standard Atmosphere, 1976, NASA-TM-X-74335, NOAA-S/T-76-1562",
    }

    return xr.Dataset(data_vars, coords, attrs)  # type: ignore


def compute_levels_temperature_and_pressure_low_altitude() -> t.Tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """Computes the temperature and the pressure values at the 8 levels
    of the low-altitude model.

    Returns
    -------
    tuple:
         Levels temperature values [K] and levels pressure values [Pa].
    """
    tb = [T0]
    pb = [P0]
    for i in range(1, 8):
        t_next = tb[i - 1] + LK[i - 1] * (H[i] - H[i - 1])
        tb.append(t_next)
        if LK[i - 1] == 0:
            p_next = compute_pressure_low_altitude_zero_gradient(
                H[i], H[i - 1], pb[i - 1], tb[i - 1]
            )
        else:
            p_next = compute_pressure_low_altitude_non_zero_gradient(
                H[i], H[i - 1], pb[i - 1], tb[i - 1], LK[i - 1]
            )
        pb.append(p_next)
    return np.array(tb), np.array(pb)


@ureg.wraps(ret=None, args="km", strict=False)
def compute_number_densities_high_altitude(
    altitudes: t.Union[ureg.Quantity, npt.NDArray[np.float64]]
) -> t.Dict[str, npt.NDArray[np.float64]]:
    """Computes the number density of the individual species in the
    high-altitude region.

    .. note::
        A uniform altitude grid is generated and used for the computation of the
        integral as well as for the computation of the number densities of the
        individual species. This gridded data is then interpolated at the query
        ``altitudes`` using a linear interpolation scheme in logarithmic space.

    Parameters
    ----------
    altitudes: :class:`~ureg.Quantity`
        Altitude value(s) [km].

    Returns
    -------
    :class:`~numpy.ndarray`
        Number densities of the individual species and total number density at
        the given altitudes [m^-3].
        The number densities of the individual species are stored in a single
        2-D array where the first dimension is the gas species and the second
        dimension is the altitude. The species are in the following order:
        =====  ======
        Row    Species
        =====  ======
         0       N2
         1       O
         2       O2
         3       Ar
         4       He
         5       H
        =====  ======
    """

    # altitude grid
    grid = np.concatenate(  # type: ignore
        (
            np.linspace(start=Z7, stop=150.0, num=640, endpoint=False),
            np.geomspace(start=150.0, stop=Z12, num=100, endpoint=True),
        )
    )  # [km]

    # pre-computed variables
    m = compute_mean_molar_mass_high_altitude(grid)  # [kg/mol]
    g = compute_gravity(ureg.Quantity(grid, "km"))  # [m / s^2]
    t = compute_temperature_high_altitude(grid)  # [K]
    dt_dz = compute_temperature_gradient_high_altitude(grid)  # [K/m]
    below_115 = grid < 115.0
    k = eddy_diffusion_coefficient(grid[below_115])  # [m^2/s]

    n_grid = {}

    # molecular nitrogen
    y = m * g / (R * t)  # [m^-1]
    n_grid["N2"] = (
        N2_7 * (T7 / t) * np.exp(-cumtrapz(y, 1e3 * grid, initial=0.0))
    )  # the factor 1000 is to convert km to m

    # atomic oxygen
    d = thermal_diffusion_coefficient(
        background=n_grid["N2"][below_115], temperature=t[below_115], a=A["O"], b=B["O"]
    )
    y = thermal_diffusion_term_atomic_oxygen(
        grid, g, t, dt_dz, d, k
    ) + velocity_term_atomic_oxygen(grid)
    n_grid["O"] = O_7 * (T7 / t) * np.exp(-cumtrapz(y, 1e3 * grid, initial=0.0))

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
    n_grid["O2"] = O2_7 * (T7 / t) * np.exp(-cumtrapz(y, 1e3 * grid, initial=0.0))

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
    n_grid["Ar"] = AR_7 * (T7 / t) * np.exp(-cumtrapz(y, 1e3 * grid, initial=0.0))

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
    n_grid["He"] = HE_7 * (T7 / t) * np.exp(-cumtrapz(y, 1e3 * grid, initial=0.0))

    # hydrogen

    # below 500 km
    mask = (grid >= 150.0) & (grid <= 500.0)
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
    integral_values = cumtrapz(y[::-1], 1e3 * grid[mask][::-1], initial=0.0)
    integral_values = integral_values[::-1]
    n_below_500 = (
        (H_11 - integral_values) * np.power(T11 / t[mask], 1 + alpha) * np.exp(-_tau)
    )

    # above 500 km
    _tau = tau_function(grid[grid > 500.0], below_500=False)
    n_above_500 = H_11 * np.power(T11 / t[grid > 500.0], 1 + alpha) * np.exp(-_tau)

    n_grid["H"] = np.concatenate((n_below_500, n_above_500))  # type: ignore

    n = {
        s: log_interp1d(grid, n_grid[s])(altitudes)
        for s in ["N2", "O", "O2", "Ar", "He"]
    }

    # Below 150 km, the number density of atomic hydrogen is zero.
    n_h_below_150 = np.zeros(len(altitudes[altitudes < 150.0]))
    n_h_above_150 = log_interp1d(grid[grid >= 150.0], n_grid["H"])(
        altitudes[altitudes >= 150.0]
    )
    n["H"] = np.concatenate((n_h_below_150, n_h_above_150))  # type: ignore

    return n


@ureg.wraps(ret=None, args="km", strict=False)
def compute_mean_molar_mass_high_altitude(
    z: t.Union[ureg.Quantity, npt.NDArray[np.float64], float]
) -> npt.NDArray[np.float64]:
    """Computes the mean molar mass in the high-altitude region.

    Parameters
    ----------
    z: :class:`~numpy.ndarray`
        Altitude [km].

    Returns
    -------
    :class:`~numpy.ndarray`
        Mean molar mass [kg/mol].
    """
    return np.array(np.where(z <= 100.0, M0, M["N2"]))


@ureg.wraps(ret=None, args="km", strict=False)
def compute_temperature_high_altitude(
    altitude: t.Union[ureg.Quantity, npt.NDArray[np.float64], float]
) -> npt.NDArray[np.float64]:
    """Computes the temperature in the high-altitude region.

    Parameters
    ----------
    altitude: :class:`~numpy.ndarray`
        Altitude values [km].

    Returns
    -------
    :class:`~numpy.ndarray`
        Temperature values [K].
    """
    r0 = R0 / 1e3  # km
    a = -76.3232  # K
    b = -19.9429  # km
    tc = 263.1905  # K

    def t(z: float) -> float:
        r"""Compute the temperature at a given altitude.

        Parameters
        ----------
        z: float
            Altitude [km].

        Returns
        -------
        float
            Temperature [K].
        """
        if Z7 <= z <= Z8:
            return T7
        elif Z8 < z <= Z9:
            return tc + a * float(np.sqrt(1.0 - np.power((z - Z8) / b, 2.0)))
        elif Z9 < z <= Z10:
            return T9 + LK9 * (z - Z9)
        elif Z10 < z <= Z12:
            return TINF - (TINF - T10) * float(
                np.exp(-LAMBDA * (z - Z10) * (r0 + Z10) / (r0 + z))
            )
        else:
            raise ValueError("altitude value is out of range")

    return np.array(np.vectorize(t)(altitude))


@ureg.wraps(ret=None, args="km", strict=False)
def compute_temperature_gradient_high_altitude(
    altitude: t.Union[ureg.Quantity, npt.NDArray[np.float64]]
) -> npt.NDArray[np.float64]:
    """Computes the temperature gradient in the high-altitude region.

    Parameters
    ----------
    altitude: :class:`~ureg.Quantity`
        Altitude values [km].

    Returns
    -------
    :class:`np.ndarray`
        Temperature gradient values [K/m].
    """
    a = -76.3232  # [dimensionless]
    b = -19.9429  # km

    def gradient(z: float) -> float:
        """Computes the temperature gradient at a given altitude.

        Parameters
        ----------
        z: float
            Altitude [km].

        Returns
        -------
        float
            Temperature gradient [K/km].
        """
        if Z7 <= z <= Z8:
            return LK7
        elif Z8 < z <= Z9:
            return -a / b * ((z - Z8) / b) / float(np.sqrt(1 - np.square((z - Z8) / b)))
        elif Z9 < z <= Z10:
            return LK9
        elif Z10 < z <= Z12:
            zeta = (z - Z10) * (R0 + Z10) / (R0 + z)
            return (
                LAMBDA
                * (TINF - T10)
                * float(np.square((R0 + Z10) / (R0 + z)))
                * float(np.exp(-LAMBDA * zeta))
            )

        else:
            raise ValueError(f"altitude z out of range, should be in " f"[{Z7}, {Z12}]")

    return np.array(np.vectorize(gradient)(altitude) / 1e3)  # converts K/km to K/m


@ureg.wraps(ret=None, args=("m^-3", "K", "m^-1*s^-1", "m^-1*s^-1"), strict=False)
def thermal_diffusion_coefficient(
    background: t.Union[ureg.Quantity, npt.NDArray[np.float64]],
    temperature: t.Union[ureg.Quantity, npt.NDArray[np.float64]],
    a: t.Union[ureg.Quantity, float],
    b: t.Union[ureg.Quantity, float],
) -> npt.NDArray[np.float64]:
    """Computes the thermal diffusion coefficient values in the
    high-altitude region.

    Parameters
    ----------
    n: :class:`numpy.ndarray`
        Background number density values [m^-3].

    t: :class:`numpy.ndarray`
        Temperature values [K].

    a: float
        Thermal diffusion constant a [m^-1*s^-1].

    b: float
        Thermal diffusion constant b [m^-1*s^-1].

    Returns
    -------
    :class:`~numpy.ndarray`
        Values of the thermal diffusion coefficient [m^2/s].
    """
    return np.array((a / background) * np.power(temperature / 273.15, b))


@ureg.wraps(ret=None, args="km", strict=False)
def eddy_diffusion_coefficient(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Computes the values of the Eddy diffusion coefficient in the
    high-altitude region.

    .. note::
        Valid in the altitude region :math:`86 <= z <= 150` km.

    Parameters
    ----------
    z: :class:`~numpy.ndarray`
        Altitude values [km].

    Returns
    -------
    :class:`~numpy.ndarray`
        Eddy diffusion coefficient values [m^2/s].
    """
    return np.array(
        np.where(
            z < 95.0, K_7, K_7 * np.exp(1.0 - (400.0 / (400.0 - np.square(z - 95.0))))
        )
    )


@ureg.wraps(
    ret=None,
    args=("m/s^2", "K", "K/m", "kg/mol", "kg/mol", None, "m^2/s", "m^2/s"),
    strict=False,
)
def f_below_115_km(
    g: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    dt_dz: npt.NDArray[np.float64],
    m: t.Union[npt.NDArray[np.float64], float],
    mi: float,
    alpha: float,
    d: npt.NDArray[np.float64],
    k: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Evaluates function f below 115 km altitude.

    Evaluates the function :math:`f` defined by equation (36) in
    :cite:`NASA1976USStandardAtmosphere` in the altitude region :math:`86
    <= z <= 115` km.

    Parameters
    ----------
    g: float or array-like:
        Values of gravity at the different altitudes [m / s^2].

    t: float or array-like
        Values of temperature at the different altitudes [K].

    dt_dz: float or array-like
        Values of temperature gradient at the different altitudes [K/m].

    m: float
        Molar mass [kg/mol].

    mi: float
        Species molar mass [kg/mol]

    alpha: float
        Alpha thermal diffusion constant [dimensionless].

    d: float or array-like
        Values of the thermal diffusion coefficient at the different altitudes
        [m^2/s].

    k: float or array-like
        Values of the Eddy diffusion coefficient at the different altitudes
        [m^2/s].

    Returns
    -------
    :class:`~numpy.ndarray`
        Values of the function f at the different altitudes [m^-1].
    """
    return np.array(
        (g / (R * t)) * (d / (d + k)) * (mi + (m * k) / d + (alpha * R * dt_dz) / g)  # type: ignore
    )


@ureg.wraps(ret=None, args=("m/s^2", "K", "K/m", "kg/mol", None), strict=False)
def f_above_115_km(
    g: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    dt_dz: npt.NDArray[np.float64],
    mi: float,
    alpha: float,
) -> npt.NDArray[np.float64]:
    """Evaluates the function :math:`f` defined by equation (36) in
    :cite:`NASA1976USStandardAtmosphere` in the altitude region :math:`115 <
    z <= 1000` km.

    Parameters
    ----------
    g: float or array-like
        Values of gravity at the different altitudes [m/s^2].

    t: float or array-like
        Values of temperature at the different altitudes [K].

    dt_dz: float or array-like
        Values of temperature gradient at the different altitudes [K/m].

    mi: float
        Species molar mass [kg/mol].

    alpha: float
        Alpha thermal diffusion constant [dimensionless].

    Returns
    -------
    :class:`~numpy.ndarray`
        Values of the function f at the different altitudes [m^-1].
    """
    return np.array((g / (R * t)) * (mi + ((alpha * R) / g) * dt_dz))


@ureg.wraps(
    ret=None,
    args=(None, "km", "m/s^2", "K", "K/m", "kg/mol", "m^2/s", "m^2/s"),
    strict=False,
)
def thermal_diffusion_term(
    species: str,
    grid: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    dt_dz: npt.NDArray[np.float64],
    m: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64],
    k: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Computes the thermal diffusion term of a given species in the
    high-altitude region.

    Parameters
    ----------
    species: str
        Species.

    grid: :class:`~numpy.ndarray`
        Altitude grid [km].

    g: :class:`~numpy.ndarray`
        Values of the gravity on the altitude grid [m/s^2].

    t: :class:`~numpy.ndarray`
        Values of the temperature on the altitude grid [K].

    dt_dz: :class:`~numpy.ndarray`
        Values of the temperature gradient on the altitude grid [K/m].

    m: :class:`~numpy.ndarray`
        Values of the mean molar mass on the altitude grid [kg/mol].

    d: :class:`~numpy.ndarray`
        Values of the molecular diffusion coefficient on the altitude grid, for altitudes < 115 km [m^2/s].

    k: :class:`~numpy.ndarray`
        Values of the eddy diffusion coefficient on the altitude grid, for altitudes < 115 km [m^2/s].

    Returns
    -------
    :class:`~numpy.ndarray`
        Values of the thermal diffusion term [km^-1].
    """
    fo1 = f_below_115_km(
        g[grid < 115.0],
        t[grid < 115.0],
        dt_dz[grid < 115.0],
        m[grid < 115.0],
        M[species],
        ALPHA[species],
        d,
        k,
    )
    fo2 = f_above_115_km(
        g[grid >= 115.0],
        t[grid >= 115.0],
        dt_dz[grid >= 115.0],
        M[species],
        ALPHA[species],
    )
    fo_concat = np.concatenate((fo1, fo2))  # type: ignore
    return np.array(fo_concat)


@ureg.wraps(ret=None, args=("km", "m/s^2", "K", "K/m", "m^2/s", "m^2/s"), strict=False)
def thermal_diffusion_term_atomic_oxygen(
    grid: npt.NDArray[np.float64],
    g: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    dt_dz: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64],
    k: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Computes the thermal diffusion term of atomic oxygen in the
    high-altitude region.

    Parameters
    ----------
    grid: :class:`~numpy.ndarray`
        Altitude grid [km].

    g: :class:`~numpy.ndarray`
        Values of the gravity on the altitude grid [m/s^2].

    t: :class:`~numpy.ndarray`
        Values of the temperature on the altitude grid [K].

    dt_dz: :class:`~numpy.ndarray`
        Values of the temperature gradient on the altitude grid [K/m].

    d: :class:`~numpy.ndarray`
        Values of thermal diffusion coefficient on the altitude grid [m^2/s].

    k: :class:`~numpy.ndarray`
        Values of the Eddy diffusion coefficient on the altitude grid [m^2/s].

    Returns
    -------
    :class:`~numpy.ndarray`
        Values of the thermal diffusion term [km^-1].
    """
    mask1, mask2 = grid < 115.0, grid >= 115.0  # type: ignore
    x1 = f_below_115_km(
        g[mask1], t[mask1], dt_dz[mask1], M["N2"], M["O"], ALPHA["O"], d, k
    )
    x2 = f_above_115_km(
        g[grid >= 115.0], t[grid >= 115.0], dt_dz[grid >= 115.0], M["O"], ALPHA["O"]
    )
    x_concat = np.concatenate((x1, x2))  # type: ignore
    return np.array(x_concat)


@ureg.wraps(
    ret=None, args=("m", "km^-3", "km^-3", "km", "km", "km^-3", "km^-3"), strict=False
)
def velocity_term_hump(
    z: npt.NDArray[np.float64],
    q1: float,
    q2: float,
    u1: float,
    u2: float,
    w1: float,
    w2: float,
) -> npt.NDArray[np.float64]:
    """Computes the transport term given by equation (37) in
    :cite:`NASA1976USStandardAtmosphere`.

    .. note::
        Valid in the altitude region: 86 km <= z <= 150 km

    Parameters
    ----------
    z: :class:`~numpy.ndarray` or float
        Altitude [km].

    q1: float
        Value of the Q constant [km^-3].

    q2: float
        Value of the q constant [km^-3].

    u1: float
        Value of the U constant [km].

    u2: float
        Value of the u constant [km].

    w1: float
        Value of the W constant [km^-3].

    w2: float
        Value of the w constant [km^-3].

    Returns → float or array-like:
        Values of the transport term [km^-1].
    """
    # @formatter:off
    return np.array(
        (
            q1 * np.square(z - u1) * np.exp(-w1 * np.power(z - u1, 3.0))
            + q2 * np.square(u2 - z) * np.exp(-w2 * np.power(u2 - z, 3.0))
        )
        / 1e3
    )  # the factor 1e3 converts m^-1 to km^-1
    # @formatter:on


@ureg.wraps(ret=None, args=("km", "km^-3", "km^-3", "km^-3"), strict=False)
def velocity_term_no_hump(
    z: npt.NDArray[np.float64], q1: float, u1: float, w1: float
) -> npt.NDArray[np.float64]:
    """Computes the transport term given by equation (37) in
    :cite:`NASA1976USStandardAtmosphere` where the second term is zero.

    .. note::
        Valid in the altitude region :math:`86 <= z <= 150` km.

    Parameters
    ----------
    z: :class:`~numpy.ndarray` or float
        Altitude [km].

    q1: float
        Value of the Q constant [km^-3].

    u1: float
        Value of the U constant [km].

    w1: float
        Value of the W constant [km^-3].

    Returns
    -------
    float or :class:`~numpy.ndarray`:
        Values of the transport term [km^-1].
    """
    # @formatter:off
    return np.array(
        (q1 * np.square(z - u1) * np.exp(-w1 * np.power(z - u1, 3.0))) / 1e3
    )  # the factor 1e3 converts m^-1 to km^-1
    # @formatter:on


@ureg.wraps(ret=None, args=(None, "km"), strict=False)
def velocity_term(
    species: str, grid: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""Computes the velocity term of a given species in the
    high-altitude region.

    .. note::
        Not valid for atomic oxygen. See :func:`velocity_term_atomic_oxygen`

    Parameters
    ----------
    species: str
        Species.

    grid: :class:`~numpy.ndarray`
        Altitude grid [km].

    Returns
    -------
    :class:`~numpy.ndarray`
        Values of the velocity terms [km^-1].
    """
    x1 = velocity_term_no_hump(
        grid[grid <= 150.0], Q1[species], U1[species], W1[species]
    )

    # Above 150 km, the velocity term is neglected, as indicated at p. 14 in
    # :cite:`NASA1976USStandardAtmosphere`
    x2 = np.zeros(len(grid[grid > 150.0]))
    x_concat = np.concatenate((x1, x2))  # type: ignore
    return np.array(x_concat)


@ureg.wraps(ret=None, args="km", strict=False)
def velocity_term_atomic_oxygen(
    grid: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Computes the velocity term of atomic oxygen in the high-altitude region.

    Parameters
    ----------
    grid: :class:`~numpy.ndarray`
        Altitude grid [km].

    Returns
    -------
    :class:`~numpy.ndarray`
        Values of the velocity term [km^-1].
    """
    mask1, mask2 = grid <= 150.0, grid > 150.0  # type: ignore
    x1 = np.where(
        grid[mask1] <= 97.0,
        velocity_term_hump(
            grid[mask1], Q1["O"], Q2["O"], U1["O"], U2["O"], W1["O"], W2["O"]
        ),
        velocity_term_no_hump(grid[mask1], Q1["O"], U1["O"], W1["O"]),
    )

    x2 = np.zeros(len(grid[mask2]))
    x_concat = np.concatenate((x1, x2))  # type: ignore
    return np.array(x_concat)


@ureg.wraps(ret=None, args=("km", None), strict=False)
def tau_function(
    z_grid: t.Union[ureg.Quantity, npt.NDArray[np.float64]], below_500: bool = True
) -> npt.NDArray[np.float64]:
    """Computes the integral given by equation (40) in
    :cite:`NASA1976USStandardAtmosphere` at each point of an altitude grid.

    .. note::
        Valid for altitudes between 150 km and 500 km.

    Parameters
    ----------
    z_grid: :class:`~ureg.Quantity` or :class:`~numpy.ndarray`
        Altitude grid (values sorted by ascending order) to use for integration [km].

    below_500: bool
        True if altitudes in z_grid are lower than 500 km, False otherwise.

    Returns
    -------
    :class:`~numpy.ndarray`
        Integral evaluations [dimensionless].
    """
    if below_500:
        z_grid = z_grid[::-1]

    y = (
        M["H"]
        * compute_gravity(ureg.Quantity(z_grid, "km"))
        / (R * compute_temperature_high_altitude(z_grid))
    )  # [m^-1]
    integral_values = cumtrapz(y, 1e3 * z_grid, initial=0.0)  # the factor 1e3
    # converts z_grid to meters

    if below_500:
        return np.array(integral_values[::-1])
    else:
        return np.array(integral_values)


def log_interp1d(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> t.Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Computes the linear interpolation of :math:`y(x)` in logarithmic space.

    Parameters
    ----------
    x: :class:`~numpy.ndarray`
        1-D array of real value.

    y: :class:`~numpy.ndarray`
        N-D array of real values. The length of y along the interpolation axis
        must be equal to the length of x.

    Returns
    -------
    Callable
        Function whose call method uses interpolation to find the value of new
        points.
    """
    logx = np.log10(x)
    logy = np.log10(y)
    lin_interp = interp1d(logx, logy, kind="linear")

    def log_interp(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.array(np.power(10.0, lin_interp(np.log10(z))))

    return log_interp


@ureg.wraps(ret=None, args=("m", "Pa", "K"), strict=False)
def compute_pressure_low_altitude(
    h: t.Union[ureg.Quantity, npt.NDArray[np.float64]],
    pb: t.Union[ureg.Quantity, npt.NDArray[np.float64]],
    tb: t.Union[ureg.Quantity, npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Computes the pressure in the low-altitude region.

    Parameters
    ----------
    h: :class:`~ureg.Quantity` or :class:`~numpy.ndarray`
        Geopotential height values [m].

    pb: :class:`~ureg.Quantity` or :class:`~numpy.ndarray`
        Levels pressure [Pa].

    tb: :class:`~ureg.Quantity` or :class:`~numpy.ndarray`
        Levels temperature [K].

    Returns
    -------
    :class:`~numpy.ndarray`
        Pressure values [Pa].
    """
    # we create a mask for each layer
    masks = [ma.masked_inside(h, H[i - 1], H[i]).mask for i in range(1, 8)]  # type: ignore

    # for each layer, we evaluate the pressure based on whether the
    # temperature gradient is zero or not
    p = np.empty(len(h))
    for i, mask in enumerate(masks):
        if LK[i] == 0:
            p[mask] = compute_pressure_low_altitude_zero_gradient(
                h[mask], H[i], pb[i], tb[i]
            )
        else:
            p[mask] = compute_pressure_low_altitude_non_zero_gradient(
                h[mask], H[i], pb[i], tb[i], LK[i]
            )
    return p


@ureg.wraps(ret=None, args=("m", "m", "Pa", "K"), strict=False)
def compute_pressure_low_altitude_zero_gradient(
    h: t.Union[ureg.Quantity, npt.NDArray[np.float64], float],
    hb: t.Union[ureg.Quantity, npt.NDArray[np.float64], float],
    pb: t.Union[ureg.Quantity, npt.NDArray[np.float64], float],
    tb: t.Union[ureg.Quantity, npt.NDArray[np.float64], float],
) -> npt.NDArray[np.float64]:
    """Computes the pressure in the low-altitude region when the temperature
    gradient is zero.

    Parameters
    ----------
    h: :class:`~ureg.Quantity` or float
        Geopotential height [m].

    hb: :class:`~ureg.Quantity` or float
        Geopotential height at the bottom of the layer [m].

    pb: :class:`~ureg.Quantity` or float
        Pressure at the bottom of the layer [Pa].

    tb: :class:`~ureg.Quantity` or float
        Temperature at the bottom of the layer [K].

    Returns
    -------
    :class:`~ureg.Quantity` or float
        Pressure [Pa].
    """
    return np.array(pb * np.exp(-G0 * M0 * (h - hb) / (R * tb)))


@ureg.wraps(ret=None, args=("m", "m", "Pa", "K", "K/m"), strict=False)
def compute_pressure_low_altitude_non_zero_gradient(
    h: t.Union[ureg.Quantity, npt.NDArray[np.float64], float],
    hb: t.Union[ureg.Quantity, npt.NDArray[np.float64], float],
    pb: t.Union[ureg.Quantity, npt.NDArray[np.float64], float],
    tb: t.Union[ureg.Quantity, npt.NDArray[np.float64], float],
    lkb: t.Union[ureg.Quantity, npt.NDArray[np.float64], float],
) -> npt.NDArray[np.float64]:
    """Computes the pressure in the low-altitude region when the temperature
    gradient is non-zero.

    Parameters
    ----------
    h: :class:`~ureg.Quantity` or float
        Geopotential height [m].

    hb: :class:`~ureg.Quantity` or float
        Geopotential height at the bottom of the layer [m].

    pb: :class:`~ureg.Quantity` or float
        Pressure at the bottom of the layer [Pa].

    tb: :class:`~ureg.Quantity` or float
        Temperature at the bottom of the layer [K].

    Returns
    -------
    :class:`~ureg.Quantity` or float
        Pressure [Pa].
    """
    return np.array(pb * np.power(tb / (tb + lkb * (h - hb)), G0 * M0 / (R * lkb)))


@ureg.wraps(ret=None, args=("m", "K"), strict=False)
def compute_temperature_low_altitude(
    h: t.Union[pint.Quantity, npt.NDArray[np.float64]],
    tb: t.Union[pint.Quantity, npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Computes the temperature in the low-altitude region.

    Parameters
    ----------
    h: :class:`~numpy.ndarray`
        Geopotential height values [m].

    tb: :class:`~numpy.ndarray`
        Levels temperature values [K].

    Returns
    -------
    :class:`~numpy.ndarray`
        Temperature [K].
    """
    # we create a mask for each layer
    masks = [
        ma.masked_inside(h, H[i - 1], H[i]).mask for i in range(1, 8)  # type: ignore
    ]

    # for each layer, we evaluate the pressure based on whether the
    # temperature gradient is zero or not
    t = np.empty(len(h))
    for i, mask in enumerate(masks):
        if LK[i] == 0:
            t[mask] = tb[i]
        else:
            t[mask] = tb[i] + LK[i] * (h[mask] - H[i])
    return t


@ureg.wraps(ret=None, args="m", strict=False)
def to_altitude(
    h: t.Union[pint.Quantity, npt.NDArray[np.float64]]
) -> npt.NDArray[np.float64]:
    """Converts geopotential height to (geometric) altitude.

    Parameters
    ----------
    h: :class:`~numpy.ndarray` or float
        Geopotential altitude [m].

    Returns
    -------
    :class:`~numpy.ndarray` or float
        Altitude [m]
    """
    return np.array(R0 * h / (R0 - h))


@ureg.wraps(ret=None, args="m", strict=False)
def to_geopotential_height(
    z: t.Union[pint.Quantity, npt.NDArray[np.float64]]
) -> npt.NDArray[np.float64]:
    """Converts altitude to geopotential height.

    Parameters
    ----------
    z: :class:`~numpy.ndarray` or float
        Altitude [m].

    Returns
    -------
    :class:`~numpy.ndarray` or float
        Geopotential height [m]
    """
    return np.array(R0 * z / (R0 + z))


@ureg.wraps(ret=None, args="m", strict=False)
def compute_gravity(
    z: t.Union[pint.Quantity, npt.NDArray[np.float64]]
) -> npt.NDArray[np.float64]:
    """Computes the gravity.

    Parameters
    ----------
    z: :class:`~numpy.ndarray` or float
        Altitude [m].

    Returns
    -------
    :class:`~numpy.ndarray` or float
        Gravity [m/s^2].
    """
    return np.array(G0 * np.power((R0 / (R0 + z)), 2.0))
