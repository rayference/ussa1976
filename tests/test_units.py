"""Test cases for the units module."""
import numpy as np
import pytest
import xarray as xr

from ussa1976.units import to_quantity, ureg


@pytest.fixture
def test_dataset() -> xr.Dataset:
    """Test dataset fixture."""
    return xr.Dataset(
        data_vars={
            "p": ("z", np.random.random(50), dict(units="Pa")),
            "t": ("z", np.random.random(50)),
        },
        coords={"z": ("z", np.linspace(0, 100, 50), dict(units="m"))},
    )


def test_to_quantity(test_dataset: xr.Dataset) -> None:
    """Returns a pint.Quantity object."""
    p = to_quantity(test_dataset.p)
    assert isinstance(p, ureg.Quantity)


def test_to_quantity_no_units(test_dataset: xr.Dataset) -> None:
    """Raises when data variable has no units."""
    with pytest.raises(ValueError):
        to_quantity(test_dataset.t)
