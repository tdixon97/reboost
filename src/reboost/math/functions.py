from __future__ import annotations

import logging

import awkward as ak
from lgdo import Array, VectorOfVectors
from lgdo.types import LGDO

log = logging.getLogger(__name__)


def piecewise_linear_activeness(
    distances: VectorOfVectors | ak.Array, fccd: float, tl: float
) -> VectorOfVectors:
    r""" Piecewise linear HPGe activeness model.

    Based on:

    .. math::

        f(d) =
        \begin{cases}
        0 & \text{if } d < t, \\
        \frac{x-l}{f - l} & \text{if } t \leq d < f, \\
        1 & \text{otherwise.}
        \end{cases}

    Where:
    - `d`: Distance to surface,
    - `l`: Depth of transition layer start
    - `f`: Full charge collection depth (FCCD).

    Parameters
    ----------
    distances
        the distance from each step to the detector surface. Can be either a
        `numpy` or `akward` array, or a LGDO `VectorOfVectors` or `Array`. The computation
        is performed for each element and the shape preserved in the output.

    fccd
        the value of the FCCD
    tl
        the start of the transition layer.

    Returns
    -------
    a :class:`LGDO.VectorOfVectors` or `LGDO.Array` of the activeness

    """

    # convert to ak
    if isinstance(distances, LGDO):
        distances_ak = distances.view_as("ak")
    elif not isinstance(distances, ak.Array):
        distances_ak = ak.Array(distances)
    else:
        distances_ak = distances

    # compute the linear piecewise
    results = ak.where(
        distances_ak > fccd, 1, ak.where(distances_ak <= tl, 0, (distances_ak - tl) / (fccd - tl))
    )
    return VectorOfVectors(results) if results.ndim > 1 else Array(results.to_numpy())
