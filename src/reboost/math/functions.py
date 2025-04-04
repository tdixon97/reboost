from __future__ import annotations

import logging

import awkward as ak
import numpy as np
from lgdo import Array, VectorOfVectors
from lgdo.types import LGDO

log = logging.getLogger(__name__)


def piecewise_linear_activeness(
    distances: VectorOfVectors | ak.Array, fccd: float, dlf: float
) -> VectorOfVectors | Array:
    r"""Piecewise linear HPGe activeness model.

    Based on:

    .. math::

        f(d) =
        \begin{cases}
        0 & \text{if } d < f*l, \\
        \frac{x-f*l}{f - f*l} & \text{if } t \leq d < f, \\
        1 & \text{otherwise.}
        \end{cases}

    Where:

    - `d`: Distance to surface,
    - `l`: Dead layer fraction, the fraction of the FCCD which is fully inactive
    - `f`: Full charge collection depth (FCCD).

    In addition, any distance of `np.nan` (for example if the calculation
    was not performed for some steps) is assigned an activeness of one.

    Parameters
    ----------
    distances
        the distance from each step to the detector surface. Can be either a
        `numpy` or `awkward` array, or a LGDO `VectorOfVectors` or `Array`. The computation
        is performed for each element and the shape preserved in the output.

    fccd
        the value of the FCCD
    dlf
        the fraction of the FCCD which is fully inactive.

    Returns
    -------
    a :class:`VectorOfVectors` or :class:`Array` of the activeness
    """
    # convert to ak
    if isinstance(distances, LGDO):
        distances_ak = distances.view_as("ak")
    elif not isinstance(distances, ak.Array):
        distances_ak = ak.Array(distances)
    else:
        distances_ak = distances

    dl = fccd * dlf

    # compute the linear piecewise
    results = ak.where(
        (distances_ak > fccd) | np.isnan(distances_ak),
        1,
        ak.where(distances_ak <= dl, 0, (distances_ak - dl) / (fccd - dl)),
    )
    return VectorOfVectors(results) if results.ndim > 1 else Array(results.to_numpy())


def vectorised_active_energy(
    distances: VectorOfVectors | ak.Array,
    edep: VectorOfVectors | ak.Array,
    fccd: float | list,
    dlf: float | list,
) -> VectorOfVectors | Array:
    r"""Energy after piecewise linear HPGe activeness model vectorised over FCCD or dead layer fraction.

    Based on the same linear activeness function as :func:`piecewise_linear_activeness`. However,
    this function vectorises the calculation to provide a range of output energies varying the fccd or
    dead layer fraction. Either fccd or dlf can be a list. This adds an extra dimension to the
    output.

    > *Warning*
    > It is not currently implemented to vary both dlf and fccd.

    Parameters
    ----------
    distances
        the distance from each step to the detector surface. Can be either a
        `numpy` or `awkward` array, or a LGDO `VectorOfVectors` or `Array`. The computation
        is performed for each element and the shape preserved in the output.
    edep
        the energy for each step.
    fccd
        the value of the FCCD, can
    dlf
        the fraction of the FCCD which is fully inactive.

    Returns
    -------
    a :class:`VectorOfVectors` or :class:`Array` of the activeness
    """
    # add checks on fccd, dlf
    fccd = np.array(fccd)
    dlf = np.array(dlf)

    if (fccd.ndim + dlf.ndim) > 1:
        msg = "Currently only one of FCCD and dlf can be varied"
        raise NotImplementedError(msg)

    dl = fccd * dlf

    def _convert(field):
        # convert to ak
        if isinstance(field, LGDO):
            field_ak = field.view_as("ak")
        elif not isinstance(field, ak.Array):
            field_ak = ak.Array(field)
        else:
            field_ak = field

        return field_ak, ak.flatten(field_ak).to_numpy()[:, np.newaxis]

    distances_ak, distances_flat = _convert(distances)
    _, edep_flat = _convert(edep)
    runs = ak.num(distances_ak, axis=-1)

    # vectorise fccd or tl
    fccd_list = np.tile(fccd, (len(distances_flat), 1))
    dl_list = np.tile(dl, (len(distances_flat), 1))

    # compute the linear piecewise
    results = ak.where(
        (distances_flat > fccd_list) | np.isnan(distances_flat),
        1,
        ak.where(distances_flat <= dl_list, 0, (distances_flat - dl_list) / (fccd_list - dl_list)),
    )
    energy = ak.sum(ak.unflatten(results * edep_flat, runs), axis=-2)

    return VectorOfVectors(energy) if energy.ndim > 1 else Array(energy.to_numpy())
