from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import lgdo
import numpy as np
import pint
from dbetto import AttrsDict
from lgdo import lh5
from scipy.interpolate import RegularGridInterpolator


class HPGePulseShapeLibrary(NamedTuple):
    """A set of templates defined in the cylindrical-like (r, z) HPGe plane."""

    waveforms: np.array
    "Field, function of the coordinates (r, z)."
    r_units: pint.Unit
    "Physical units of the coordinate `r`."
    z_units: pint.Unit
    "Physical units of the coordinate `z`."
    t_units: pint.Unit
    "Physical units of the times."
    r: np.array
    "One dimensional arrays specifying the radial coordinates"
    z: np.array
    "One dimensional arrays specifying the z coordinates"
    t: np.array
    "Times used to define the waveforms"


def get_hpge_pulse_shape_library(
    filename: str, obj: str, field: str, out_of_bounds_val: int | float = np.nan
) -> HPGePulseShapeLibrary:
    """Create the pulse shape library, holding simulated waveforms.

    Reads from disk the following data structure: ::

        FILENAME/
        └── OBJ · struct{r,z,dt,t0,FIELD}
            ├── r · array<1>{real} ── {'units': 'UNITS'}
            ├── z · array<1>{real} ── {'units': 'UNITS'}
            ├── dt · real ── {'units': 'UNITS'}
            ├── t0 · real ── {'units': 'UNITS'}
            └── FIELD · array<3>{real} ── {'units': 'UNITS'}

    The conventions follow those used for :func:`get_hpge_rz_field`.
    For the FIELD the first and second dimensions are `r` and `z`, respectively, with the last
    dimension representing the waveform. dt and t0 define the timestamps for the waveforms.


    Parameters
    ----------
    filename
        name of the LH5 file containing the gridded scalar field.
    obj
        name of the HDF5 dataset where the data is saved.
    field
        name of the HDF5 dataset holding the waveforms.
    out_of_bounds_val
        value to use to replace NaNs in the field values.
    """
    data = lh5.read(obj, filename)

    if not isinstance(data, lgdo.Struct):
        msg = f"{obj} in {filename} is not an LGDO Struct"
        raise ValueError(msg)

    t0 = data["t0"].view_as(with_units=True)
    dt = data["dt"].view_as(with_units=True)

    if t0.u != dt.u:
        msg = "t0 and dt must have the same units"
        raise ValueError(msg)
    tu = t0.u

    data = AttrsDict(
        {
            k: np.nan_to_num(data[k].view_as("np", with_units=True), nan=out_of_bounds_val)
            for k in ("r", "z", field)
        }
    )

    times = t0.m + np.arange(np.shape(data.waveforms.m)[2]) * dt.m

    return HPGePulseShapeLibrary(data.waveforms.m, data.r.u, data.z.u, tu, data.r, data.z, times)


class HPGeRZField(NamedTuple):
    """A field defined in the cylindrical-like (r, z) HPGe plane."""

    φ: Callable
    "Field, function of the coordinates (r, z)."
    r_units: pint.Unit
    "Physical units of the coordinate `r`."
    z_units: pint.Unit
    "Physical units of the coordinate `z`."
    φ_units: pint.Unit
    "Physical units of the field."
    ndim: int
    "Number of dimensions for the field"


def get_hpge_rz_field(
    filename: str, obj: str, field: str, out_of_bounds_val: int | float = np.nan, **kwargs
) -> HPGeRZField:
    """Create an interpolator for a gridded HPGe field defined on `(r, z)`.

    Reads from disk the following data structure: ::

        FILENAME/
        └── OBJ · struct{r,z,FIELD}
            ├── r · array<1>{real} ── {'units': 'UNITS'}
            ├── z · array<1>{real} ── {'units': 'UNITS'}
            └── FIELD · array<N+2>{real} ── {'units': 'UNITS'}

    where ``FILENAME``, ``OBJ`` and ``FIELD`` are provided as
    arguments to this function. `obj` is a :class:`~lgdo.types.struct.Struct`,
    `r` and `z` are one dimensional arrays specifying the radial and z
    coordinates of the rectangular grid — not the coordinates of each single
    grid point. In this coordinate system, the center of the p+ contact surface
    is at `(0, 0)`, with the p+ contact facing downwards. `field` is instead a
    ndim plus two-dimensional array specifying the field value at each grid point. The
    first and second dimensions are `r` and `z`, respectively, with the latter dimensions
    representing the dimensions of the output field.

    NaN values are interpreted as points outside the detector profile in the `(r, z)` plane.

    Before returning a :class:`HPGeScalarRZField`, the gridded field is fed to
    :class:`scipy.interpolate.RegularGridInterpolator`.

    Parameters
    ----------
    filename
        name of the LH5 file containing the gridded scalar field.
    obj
        name of the HDF5 dataset where the data is saved.
    field
        name of the HDF5 dataset holding the field values.
    out_of_bounds_val
        value to use to replace NaNs in the field values.
    """
    data = lh5.read(obj, filename)

    if not isinstance(data, lgdo.Struct):
        msg = f"{obj} in {filename} is not an LGDO Struct"
        raise ValueError(msg)

    data = AttrsDict(
        {
            k: np.nan_to_num(data[k].view_as("np", with_units=True), nan=out_of_bounds_val)
            for k in ("r", "z", field)
        }
    )
    ndim = data[field].m.ndim - 2
    interpolator = RegularGridInterpolator((data.r.m, data.z.m), data[field].m, **kwargs)

    return HPGeRZField(interpolator, data.r.u, data.z.u, data[field].u, ndim)
