from __future__ import annotations

from typing import Callable, NamedTuple

import lgdo
import numpy as np
import pint
from dbetto import AttrsDict
from lgdo import lh5
from scipy.interpolate import RegularGridInterpolator


class HPGeScalarRZField(NamedTuple):
    """A scalar field defined in the cylindrical-like (r, z) HPGe plane."""

    φ: Callable
    "Scalar field, function of the coordinates (r, z)."
    r_units: pint.Unit
    "Physical units of the coordinate `r`."
    z_units: pint.Unit
    "Physical units of the coordinate `z`."
    φ_units: pint.Unit
    "Physical units of the field."


def get_hpge_scalar_rz_field(
    filename: str, obj: str, field: str, out_of_bounds_val: int | float = np.nan, **kwargs
) -> HPGeScalarRZField:
    """Create an interpolator for a gridded scalar HPGe field defined on `(r, z)`.

    Reads from disk the following data structure: ::

        FILENAME/
        └── OBJ · struct{r,z,FIELD}
            ├── r · array<1>{real} ── {'units': 'UNITS'}
            ├── z · array<1>{real} ── {'units': 'UNITS'}
            └── FIELD · array<2>{real} ── {'units': 'UNITS'}

    where ``FILENAME``, ``OBJ`` and ``FIELD`` are provided as
    arguments to this function. `obj` is a :class:`~lgdo.types.struct.Struct`,
    `r` and `z` are one dimensional arrays specifying the radial and z
    coordinates of the rectangular grid — not the coordinates of each single
    grid point. In this coordinate system, the center of the p+ contact surface
    is at `(0, 0)`, with the p+ contact facing downwards. `field` is instead a
    two-dimensional array specifying the field value at each grid point. The
    first and second dimensions are `r` and `z`, respectively. NaN values are
    interpreted as points outside the detector profile in the `(r, z)` plane.

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

    interpolator = RegularGridInterpolator((data.r.m, data.z.m), data[field].m, **kwargs)

    return HPGeScalarRZField(interpolator, data.r.u, data.z.u, data[field].u)
