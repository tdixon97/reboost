from __future__ import annotations

import awkward as ak
import pint
import pyg4ometry as pg4
import pytest
from lgdo.types import Array

from reboost import units
from reboost.units import ureg as u


def test_registry():
    assert "meter" in units.ureg


def test_pg4_to_pint():
    assert units.pg4_to_pint(1 * u.m) == 1 * u.m
    assert all(
        units.pg4_to_pint(pg4.gdml.Defines.Position("boh", 1, 1, 1, unit="m")) == (1, 1, 1) * u.m
    )

    with pytest.raises(ValueError):
        units.pg4_to_pint({"a": 1})


def test_units_convfact():
    assert units.units_convfact(1, "m") == 1

    arr = Array([1, 2, 3], attrs={"units": "m"})
    assert units.units_convfact(arr, "mm") == 1000


def test_unwrap_lgdo():
    arr = Array([1, 2, 3], attrs={"units": "m"})

    arr_view, unit = units.unwrap_lgdo(arr, "ak")
    assert isinstance(arr_view, ak.Array)
    assert isinstance(unit, pint.Unit)
    assert unit == u.m

    arr_view_2, unit = units.unwrap_lgdo(arr_view, "ak")
    assert arr_view_2 is arr_view
    assert unit is None


def test_unit_to_lh5_attr():
    assert units.unit_to_lh5_attr((1 * u.m).u) == "m"
    assert units.unit_to_lh5_attr(u.m) == "m"
    assert units.unit_to_lh5_attr(u.m / u.s) == "m/s"
    assert units.unit_to_lh5_attr(u.m**2) == "m**2"
    assert units.unit_to_lh5_attr(u.micrometer) == "µm"
