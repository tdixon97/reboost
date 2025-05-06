from __future__ import annotations

import logging
from typing import Any

import pint
import pyg4ometry as pg4
from lgdo import LGDO

log = logging.getLogger(__name__)

ureg = pint.get_application_registry()
"""The physical units registry."""

# default pretty printing of physical units
ureg.formatter.default_format = "~P"


def pg4_to_pint(obj) -> pint.Quantity:
    """Convert pyg4ometry object to pint Quantity."""
    if isinstance(obj, pint.Quantity):
        return obj
    if isinstance(obj, pg4.gdml.Defines.VectorBase):
        return [getattr(obj, field).eval() for field in ("x", "y", "z")] * ureg(obj.unit)
    msg = f"I don't know how to convert object of type {type(obj)} to pint object"
    raise ValueError(msg)


def units_convfact(data: Any, target_units: pint.Units) -> float:
    """Calculate numeric conversion factor to reach `target_units`.

    Parameters
    ----------
    data
        starting data structure. If an LGDO, try to determine units by peeking
        into its attributes. Otherwise, just return 1.
    target_units
        units you wish to convert data to.
    """
    if isinstance(data, LGDO) and "units" in data.attrs:
        return ureg(data.attrs["units"]).to(target_units).magnitude
    return 1


def unwrap_lgdo(data: Any, library: str = "ak") -> tuple(Any, pint.Unit | None):
    """Return a view of the data held by the LGDO and its physical units.

    Parameters
    ----------
    data
        the data container. If not an LGDO, it will be returned as is with
        ``None`` units.
    library
        forwarded to :func:`lgdo.view_as`.

    Returns
    -------
    A tuple of the un-lgdo'd data and the data units.
    """
    ret_data = data
    ret_units = None
    if isinstance(data, LGDO):
        ret_data = data.view_as(library)
        if "units" in data.attrs:
            ret_units = ureg(data.attrs["units"]).u

    return ret_data, ret_units


def unit_to_lh5_attr(unit: pint.Unit) -> str:
    """Convert Pint unit to a string that can be used as attrs["units"] in an LGDO."""
    # TODO: we should check if this can be always parsed by Unitful.jl
    if isinstance(unit, pint.Unit):
        return f"{unit:~C}"
    return unit
