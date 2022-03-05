"""Units module."""
import pint
import xarray as xr

ureg = pint.UnitRegistry()


def to_quantity(da: xr.DataArray) -> pint.Quantity:  # type: ignore[type-arg]
    """Convert a DataArray to a quantity.

    The array's ``attrs`` metadata mapping must contain a ``units`` field.

    Parameters
    ----------
    da : DataArray
        :class:`~xarray.DataArray` instance which will be converted.

    Returns
    -------
    quantity
        The corresponding Pint quantity.

    Raises
    ------
    ValueError
        If the array's metadata do not contain a ``units`` field.

    Notes
    -----
    This function can also be used on coordinate variables.
    """
    try:
        units = da.attrs["units"]
    except KeyError as e:
        raise ValueError("this DataArray has no 'units' metadata field") from e
    else:
        return ureg.Quantity(da.values, units)
