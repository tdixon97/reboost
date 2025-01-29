from __future__ import annotations

import logging

import awkward as ak
import legendhpges
import numpy as np
from lgdo import Array, VectorOfVectors
from lgdo.types import LGDO
from numpy.typing import ArrayLike

log = logging.getLogger(__name__)


def distance_to_surface(
    positions_x: VectorOfVectors,
    positions_y: VectorOfVectors,
    positions_z: VectorOfVectors,
    hpge: legendhpges.base.HPGe,
    det_pos: ArrayLike,
    *,
    surface_type: str | None = None,
    unit: str = "mm",
    distances_precompute: VectorOfVectors | None = None,
    precompute_cutoff: float | None = None,
) -> Array:
    """Computes the distance from each step to the detector surface.

    The calculation can be performed for any surface type `nplus`, `pplus`,
    `passive` or `None`. In order to speed up the calculation we provide
    an option to only compute the distance for points within a certain distance
    of any surface (as computed by remage and stored in the "distances_precompute") argument.

    Parameters
    ----------
    positions_x
        Global x positions for each step.
    positions_y
        Global y positions for each step.
    positions_z
        Global z positions for each step.
    hpge
        HPGe object.
    det_pos
        position of the detector origin, must be a 3 component array corresponding to `(x,y,z)`.
    surface_type
        string of which surface to use, can be `nplus`, `pplus` `passive` or None (in which case the distance to any surface is calculated).
    unit
        unit for the hit tier positions table.
    distances_precompute
        VectorOfVectors of distance to any surface computed by remage.
    precompute_cutoff
        cutoff on distances_precompute to not compute the distance for (in mm)

    Returns
    -------
    VectorOfVectors with the same shape as `positions_x/y/z` of the distance to the surface.

    Note
    ----
    `positions_x/positions_y/positions_z` must all have the same shape.

    """
    factor = np.array([1, 100, 1000])[unit == np.array(["mm", "cm", "m"])][0]

    # compute local positions
    pos = []
    sizes = []

    for idx, pos_tmp in enumerate([positions_x, positions_y, positions_z]):
        local_pos_tmp = ak.Array(pos_tmp) * factor - det_pos[idx]
        local_pos_flat_tmp = ak.flatten(local_pos_tmp).to_numpy()
        pos.append(local_pos_flat_tmp)
        sizes.append(ak.num(local_pos_tmp, axis=1))

    if not ak.all(sizes[0] == sizes[1]) or not ak.all(sizes[0] == sizes[2]):
        msg = "all position vector of vector must have the same shape"
        raise ValueError(msg)

    size = sizes[0]
    # restructure the positions
    local_positions = np.vstack(pos).T

    # get indices
    surface_indices = (
        np.where(np.array(hpge.surfaces) == surface_type) if surface_type is not None else None
    )

    # distance calc itself
    if distances_precompute is None:
        distances = hpge.distance_to_surface(local_positions, surface_indices=surface_indices)
    else:
        # decide when the calculation needs to be run
        if isinstance(distances_precompute, LGDO):
            distances_precompute = distances_precompute.view_as("ak")

        distances_precompute_flat = ak.flatten(distances_precompute)
        distances = np.full_like(distances_precompute_flat.to_numpy(), np.nan, dtype=float)

        # values to compute
        indices = distances_precompute_flat < precompute_cutoff

        # compute the distances
        distances[indices] = hpge.distance_to_surface(
            local_positions[indices], surface_indices=surface_indices
        )

    return VectorOfVectors(ak.unflatten(distances, size))
