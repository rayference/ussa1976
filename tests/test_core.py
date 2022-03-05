"""Test cases for the core module."""
import numpy as np
import numpy.typing as npt
import pint
import pytest
import xarray as xr

from ussa1976.core import AR_7
from ussa1976.core import compute_high_altitude
from ussa1976.core import compute_levels_temperature_and_pressure_low_altitude
from ussa1976.core import compute_low_altitude
from ussa1976.core import compute_mean_molar_mass_high_altitude
from ussa1976.core import compute_number_densities_high_altitude
from ussa1976.core import compute_temperature_gradient_high_altitude
from ussa1976.core import compute_temperature_high_altitude
from ussa1976.core import create
from ussa1976.core import H
from ussa1976.core import init_data_set
from ussa1976.core import M
from ussa1976.core import M0
from ussa1976.core import make
from ussa1976.core import O2_7
from ussa1976.core import O_7
from ussa1976.core import SPECIES
from ussa1976.core import to_altitude
from ussa1976.core import VARIABLES
from ussa1976.units import to_quantity
from ussa1976.units import ureg


def test_make() -> None:
    """Returned data set has expected data."""
    # default constructor
    profile = make()

    assert profile["z_level"].values[0] == 0.0
    assert profile["z_level"].values[-1] == 100.0
    assert profile.dims["z_layer"] == 50
    assert profile.dims["species"] == 12

    # custom levels altitudes
    profile = make(levels=ureg.Quantity(np.linspace(2.0, 15.0, 51), "km"))

    assert profile.dims["z_layer"] == 50
    assert profile["z_level"].values[0] == 2.0
    assert profile["z_level"].values[-1] == 15.0
    assert profile.dims["species"] == 12

    # custom number of layers
    profile = make(levels=ureg.Quantity(np.linspace(0.0, 150.0, 37), "km"))

    assert profile.dims["z_layer"] == 36
    assert profile["z_level"].values[0] == 0.0
    assert profile["z_level"].values[-1] == 150.0
    assert profile.dims["species"] == 12

    profile = make(levels=ureg.Quantity(np.linspace(0.0, 80.0, 2), "km"))

    assert profile.dims["z_layer"] == 1
    assert profile["z_level"].values[0] == 0.0
    assert profile["z_level"].values[-1] == 80.0
    assert profile.dims["species"] == 12


def test_make_invalid_levels() -> None:
    """Raises a ValueError on invalid level altitudes."""
    with pytest.raises(ValueError):
        make(levels=np.linspace(-4000, 50000) * ureg.m)

    with pytest.raises(ValueError):
        make(levels=np.linspace(500.0, 5000000.0) * ureg.m)


@pytest.fixture
def test_altitudes() -> pint.Quantity:
    """Test altitudes fixture."""
    return np.linspace(0.0, 100000.0, 101) * ureg.m


def test_create(test_altitudes: pint.Quantity) -> None:
    """Creates a data set with expected data."""
    ds = create(z=test_altitudes)
    assert all([v in ds.data_vars for v in VARIABLES])

    variables = ["p", "t", "n", "n_tot"]
    ds = create(z=test_altitudes, variables=variables)

    assert len(ds.dims) == 2
    assert "z" in ds.dims
    assert "species" in ds.dims
    assert len(ds.coords) == 2
    assert np.all(to_quantity(ds.z) == test_altitudes)
    assert [s for s in ds.species] == [s for s in SPECIES]
    for var in variables:
        assert var in ds
    assert all(
        [
            x in ds.attrs
            for x in ["convention", "title", "history", "source", "references"]
        ]
    )


def test_create_invalid_variables(test_altitudes: npt.NDArray[np.float64]) -> None:
    """Raises when invalid variables are given."""
    invalid_variables = ["p", "t", "invalid", "n"]
    with pytest.raises(ValueError):
        create(z=test_altitudes, variables=invalid_variables)


def test_create_invalid_z() -> None:
    """Raises when invalid altitudes values are given."""
    with pytest.raises(ValueError):
        create(z=np.array([-5.0]) * ureg.m)

    with pytest.raises(ValueError):
        create(z=np.array([1000001.0]) * ureg.m)


def test_create_below_86_km_layers_boundary_altitudes() -> None:
    """
    Produces correct results.

    We test the computation of the atmospheric variables (pressure,
    temperature and mass density) at the level altitudes, i.e. at the model
    layer boundaries. We assert correctness by comparing their values with the
    values from the table 1 of the U.S. Standard Atmosphere 1976 document.
    """
    z = to_altitude(H)
    ds = create(z=z, variables=["p", "t", "rho"])

    level_temperature = (
        np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.87])
        * ureg.K
    )
    level_pressure = (
        np.array([101325.0, 22632.0, 5474.8, 868.01, 110.90, 66.938, 3.9564, 0.37338])
        * ureg.Pa
    )
    level_mass_density = (
        np.array(
            [
                1.225,
                0.36392,
                0.088035,
                0.013225,
                0.0014275,
                0.00086160,
                0.000064261,
                0.000006958,
            ]
        )
        * ureg.kg
        / ureg.m**3
    )

    assert np.allclose(to_quantity(ds.t), level_temperature, rtol=1e-4)
    assert np.allclose(to_quantity(ds.p), level_pressure, rtol=1e-4)
    assert np.allclose(to_quantity(ds.rho), level_mass_density, rtol=1e-3)


def test_create_below_86_km_arbitrary_altitudes() -> None:
    """
    Produces correct results.

    We test the computation of the atmospheric variables (pressure,
    temperature and mass density) at arbitrary altitudes. We assert correctness
    by comparing their values to the values from table 1 of the U.S. Standard
    Atmosphere 1976 document.
    """
    # The values below were selected arbitrarily from Table 1 of the document
    # such that there is at least one value in each of the 7 temperature
    # regions.
    h = (
        np.array(
            [
                200.0,
                1450.0,
                5250.0,
                6500.0,
                9800.0,
                17900.0,
                24800.0,
                27100.0,
                37200.0,
                40000.0,
                49400.0,
                61500.0,
                79500.0,
                84000.0,
            ]
        )
        * ureg.m
    )
    temperatures = (
        np.array(
            [
                286.850,
                278.725,
                254.025,
                245.900,
                224.450,
                216.650,
                221.450,
                223.750,
                243.210,
                251.050,
                270.650,
                241.250,
                197.650,
                188.650,
            ]
        )
        * ureg.K
    )
    pressures = (
        np.array(
            [
                98945.0,
                85076.0,
                52239.0,
                44034.0,
                27255.0,
                7624.1,
                2589.6,
                1819.4,
                408.7,
                277.52,
                81.919,
                16.456,
                0.96649,
                0.43598,
            ]
        )
        * ureg.Pa
    )
    mass_densities = (
        np.array(
            [
                1.2017,
                1.0633,
                0.71641,
                0.62384,
                0.42304,
                0.12259,
                0.040739,
                0.028328,
                0.0058542,
                0.0038510,
                0.0010544,
                0.00023764,
                0.000017035,
                0.0000080510,
            ]
        )
        * ureg.kg
        / ureg.m**3
    )

    z = to_altitude(h)
    ds = create(z=z, variables=["t", "p", "rho"])

    assert np.allclose(to_quantity(ds.t), temperatures, rtol=1e-4)
    assert np.allclose(to_quantity(ds.p), pressures, rtol=1e-4)
    assert np.allclose(to_quantity(ds.rho), mass_densities, rtol=1e-4)


def test_init_data_set() -> None:
    """Data set is initialised.

    Expected data variables are created and fill with nan values.
    Expected dimensions and coordinates are present.
    """

    def check_data_set(ds: xr.Dataset) -> None:
        """Check a data set."""
        for var in VARIABLES:
            assert var in ds
            assert np.isnan(ds[var].values).all()

        assert ds.n.values.ndim == 2
        assert all(
            ds.species.values
            == ["N2", "O2", "Ar", "CO2", "Ne", "He", "Kr", "Xe", "CH4", "H2", "O", "H"]
        )

    z1 = np.linspace(0.0, 50000.0) * ureg.m
    ds1 = init_data_set(z1)
    check_data_set(ds1)

    z2 = np.linspace(120000.0, 650000.0) * ureg.m
    ds2 = init_data_set(z2)
    check_data_set(ds2)

    z3 = np.linspace(70000.0, 100000.0) * ureg.m
    ds3 = init_data_set(z3)
    check_data_set(ds3)


def test_compute_levels_temperature_and_pressure_low_altitude() -> None:
    """Computes correct level temperature and pressure values.

    The correct values are taken from :cite:`NASA1976USStandardAtmosphere`.
    """
    tb, pb = compute_levels_temperature_and_pressure_low_altitude()

    level_temperature = (
        np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.87])
        * ureg.K
    )
    level_pressure = (
        np.array([101325.0, 22632.0, 5474.8, 868.01, 110.90, 66.938, 3.9564, 0.37338])
        * ureg.Pa
    )

    assert np.allclose(tb, level_temperature, rtol=1e-3)
    assert np.allclose(pb, level_pressure, rtol=1e-3)


def test_compute_number_density() -> None:
    """Computes correct number density values at arbitrary level altitudes.

    The correct values are taken from :cite:`NASA1976USStandardAtmosphere`
    (table VIII, p. 210-215).
    """
    # the following altitudes values are chosen arbitrarily
    altitudes = (
        np.array(
            [
                86.0,
                90.0,
                95.0,
                100.0,
                110.0,
                120.0,
                150.0,
                200.0,
                300.0,
                400.0,
                500.0,
                600.0,
                700.0,
                800.0,
                900.0,
                1000.0,
            ]
        )
        * ureg.km
    )

    mask = altitudes > 150.0 * ureg.km

    # the corresponding number density values are from NASA (1976) - U.S.
    # Standard Atmosphere, table VIII (p. 210-215)
    values = {
        "N2": np.array(
            [
                1.13e20,
                5.547e19,
                2.268e19,
                9.210e18,
                1.641e18,
                3.726e17,
                3.124e16,
                2.925e15,
                9.593e13,
                4.669e12,
                2.592e11,
                1.575e10,
                1.038e9,
                7.377e7,
                5.641e6,
                4.626e5,
            ]
        )
        / ureg.m**3,
        "O": np.array(
            [
                O_7.m_as(1 / ureg.m**3),
                2.443e17,
                4.365e17,
                4.298e17,
                2.303e17,
                9.275e16,
                1.780e16,
                4.050e15,
                5.443e14,
                9.584e13,
                1.836e13,
                3.707e12,
                7.840e11,
                1.732e11,
                3.989e10,
                9.562e9,
            ]
        )
        / ureg.m**3,
        "O2": np.array(
            [
                O2_7.m_as(1 / ureg.m**3),
                1.479e19,
                5.83e18,
                2.151e18,
                2.621e17,
                4.395e16,
                2.750e15,
                1.918e14,
                3.942e12,
                1.252e11,
                4.607e9,
                1.880e8,
                8.410e6,
                4.105e5,
                2.177e4,
                1.251e3,
            ]
        )
        / ureg.m**3,
        "Ar": np.array(
            [
                AR_7.m_as(1 / ureg.m**3),
                6.574e17,
                2.583e17,
                9.501e16,
                1.046e16,
                1.366e15,
                5.0e13,
                1.938e12,
                1.568e10,
                2.124e8,
                3.445e6,
                6.351e4,
                1.313e3,
                3.027e1,
                7.741e-1,
                2.188e-2,
            ]
        )
        / ureg.m**3,
        "He": np.array(
            [
                7.582e14,
                3.976e14,
                1.973e14,
                1.133e14,
                5.821e13,
                3.888e13,
                2.106e13,
                1.310e13,
                7.566e12,
                4.868e12,
                3.215e12,
                2.154e12,
                1.461e12,
                1.001e12,
                6.933e11,
                4.850e11,
            ]
        )
        / ureg.m**3,
        "H": np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                3.767e11,
                1.630e11,
                1.049e11,
                8.960e10,
                8.0e10,
                7.231e10,
                6.556e10,
                5.961e10,
                5.434e10,
                4.967e10,
            ]
        )
        / ureg.m**3,
    }

    n = compute_number_densities_high_altitude(altitudes=altitudes)

    assert np.allclose(to_quantity(n.sel(species="N2")), values["N2"], rtol=0.01)
    # TODO: investigate the poor relative tolerance that is achieved here
    assert np.allclose(to_quantity(n.sel(species="O")), values["O"], rtol=0.1)
    assert np.allclose(to_quantity(n.sel(species="O2")), values["O2"], rtol=0.01)
    assert np.allclose(to_quantity(n.sel(species="Ar")), values["Ar"], rtol=0.01)
    assert np.allclose(to_quantity(n.sel(species="He")), values["He"], rtol=0.01)
    assert np.allclose(
        to_quantity(n.sel(species="H"))[mask], values["H"][mask], rtol=0.01
    )


def test_compute_mean_molar_mass() -> None:
    """Computes correct mean molar mass values.

    The correct values are taken from :cite:`NASA1976USStandardAtmosphere`.
    """
    # test call with scalar altitude
    assert compute_mean_molar_mass_high_altitude(90.0 * ureg.km) == M0
    assert compute_mean_molar_mass_high_altitude(200.0 * ureg.km) == M["N2"]

    # test call with array of altitudes
    z = np.linspace(86, 1000, 915) * ureg.km
    assert np.allclose(
        compute_mean_molar_mass_high_altitude(z=z),
        np.where(z <= 100.0 * ureg.km, M0, M["N2"]),
    )


def test_compute_temperature_above_86_km() -> None:
    """Compute correct temperature values.

    The correct values are taken from :cite:`NASA1976USStandardAtmosphere`.
    """
    # single altitude
    assert np.isclose(
        compute_temperature_high_altitude(altitude=90.0 * ureg.km),
        186.87 * ureg.K,
        rtol=1e-3,
    )

    # array of altitudes
    z = np.array([100.0, 110.0, 120.0, 130.0, 200.0, 500.0]) * ureg.km
    assert np.allclose(
        compute_temperature_high_altitude(altitude=z),
        np.array([195.08, 240.00, 360.0, 469.27, 854.56, 999.24]) * ureg.K,
        rtol=1e-3,
    )


def test_compute_temperature_above_86_km_invalid_altitudes() -> None:
    """Raises when altitude is out of range."""
    with pytest.raises(ValueError):
        compute_temperature_high_altitude(altitude=10.0 * ureg.km)


def test_compute_high_altitude_no_mask() -> None:
    """Returns a Dataset."""
    z = np.linspace(86, 1000) * ureg.km
    ds = init_data_set(z=z)
    compute_high_altitude(data_set=ds, mask=None, inplace=True)
    assert isinstance(ds, xr.Dataset)


def test_compute_high_altitude_not_inplace() -> None:
    """Returns a Dataset."""
    z = np.linspace(86, 1000) * ureg.km
    ds1 = init_data_set(z=z)
    ds2 = compute_high_altitude(data_set=ds1, mask=None, inplace=False)
    assert ds1 != ds2
    assert isinstance(ds2, xr.Dataset)


def test_compute_low_altitude() -> None:
    """Returns a Dataset."""
    z = np.linspace(0, 86) * ureg.km
    ds = init_data_set(z=z)
    compute_low_altitude(data_set=ds, mask=None, inplace=True)
    assert isinstance(ds, xr.Dataset)


def test_compute_low_altitude_not_inplace() -> None:
    """Returns a Dataset."""
    z = np.linspace(0, 86) * ureg.km
    ds1 = init_data_set(z=z)
    ds2 = compute_low_altitude(data_set=ds1, mask=None, inplace=False)
    assert ds1 != ds2
    assert isinstance(ds2, xr.Dataset)


def test_compute_temperature_gradient_high_altitude() -> None:
    """Raises ValueError when altitude is out of bounds."""
    with pytest.raises(ValueError):
        z = 1300 * ureg.km
        compute_temperature_gradient_high_altitude(z=z)
