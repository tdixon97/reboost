from __future__ import annotations

import logging

import awkward as ak
import legendhpges
import numba
import numpy as np
from lgdo import VectorOfVectors
from lgdo.types import LGDO
from numpy.typing import ArrayLike
from scipy import stats

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
) -> VectorOfVectors:
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

    return VectorOfVectors(ak.unflatten(distances, size), dtype=np.float32)


@numba.njit(cache=True)
def _advance_diffusion(
    charge: np.ndarray,
    factor: float,
    recomb: float = 0,
    recomb_depth: float = 600,
    delta_x: float = 10,
):
    """Make a step of diffusion using explicit Euler scheme.

    Parameters
    ----------
    charge
        charge in each space bin up to the FCCD
    factor
        the factor of diffusion for the Euler scheme
    recomb
        the recomination probability.
    recomb_depth
        the depth of the recombination region.
    delta_x
        the width of each spatial bin.

    Returns
    -------
    a tuple of the charge distribution at the next time step and the collected charge.
    """
    charge_xp1 = np.append(charge[1:], [0])
    charge_xm1 = np.append([0], charge[:-1])

    # collected charge
    collected = factor * charge[-1]

    # charge at the next step
    charge_new = charge_xp1 * factor + charge_xm1 * factor + charge * (1 - 2 * factor)

    # correction for recombination
    charge_new[0 : int(recomb_depth / delta_x)] = (1 - recomb) * charge_new[
        0 : int(recomb_depth / delta_x)
    ]

    return charge_new, collected


@numba.njit(cache=True)
def _compute_diffusion_impl(
    init_charge: np.ndarray,
    nsteps: int,
    factor: float,
    recomb: float = 0,
    recomb_depth: float = 600,
    delta_x: float = 10,
):
    """Compute the charge collected as a function of time.

    Parameters
    ----------
    init_charge
        Initial charge distribution.
    nsteps
        Number of time steps to take.
    kwargs
        Keyword arguments to pass to :func:`_advance_diffusion`
    """
    charge = init_charge
    collected_charge = np.zeros(nsteps)

    for i in range(nsteps):
        charge, collected = _advance_diffusion(
            charge, factor=factor, recomb=recomb, recomb_depth=recomb_depth, delta_x=delta_x
        )
        collected_charge[i] = collected

    return collected_charge


def get_surface_response(
    fccd: float,
    recomb_depth: float,
    init: float = 0,
    recomb: float = 0.002,
    init_size: float = 0.0,
    factor: float = 0.29,
    nsteps: int = 10000,
    delta_x: float = 10,
):
    """Extract the surface response current pulse based on diffusion.

    Parameters
    ----------
    fccd
        the full charge collection depth (in um)
    recomb_depth
        the depth of the recombination region (in um)
    init
        the initial position of the charge (in um)
    recomb
        the recombination rate
    init_size
        the initial size of the charge cloud (in um)
    factor
        the factor for the explicit Euler scheme (the probability of charge diffusuion)
    nsteps
        the number of time steps.
    delta_x
        the width of each position bin.
    """
    # number of position steps
    nx = int(fccd / delta_x)

    # initial charge
    charge = np.zeros(nx)

    # generate initial conditions
    x = (fccd / nx) * np.arange(nx)
    x_full = (fccd / nx) * np.arange(2 * nx)

    # generate initial conditions
    if init_size != 0:
        charge = stats.norm.pdf(x, loc=init, scale=init_size)
        charge_full = stats.norm.pdf(x_full, loc=init, scale=init_size)
        charge_col = [(np.sum(charge_full) - np.sum(charge)) / np.sum(charge_full)]
        charge = charge / np.sum(charge_full)
    elif int(init * nx / fccd) < len(charge):
        charge[int(init * nx / fccd)] = 1
        charge_col = np.array([])
    else:
        charge_col = np.array([1])

    # run the simulation
    charge_collected = _compute_diffusion_impl(
        charge,
        nsteps=nsteps,
        factor=factor,
        recomb=recomb,
        recomb_depth=recomb_depth,
        delta_x=delta_x,
    )

    return np.cumsum(np.concatenate((charge_col, charge_collected)))
