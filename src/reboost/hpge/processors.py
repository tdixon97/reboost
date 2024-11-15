from __future__ import annotations

import awkward as ak
import legendhpges
import lgdo
import numpy as np
from lgdo import Array, Table, VectorOfVectors
from numpy.typing import ArrayLike


def sort_data(obj: ak.Array) -> ak.Array:
    """Sort the data by evtid then time.

    Parameters
    ----------
    obj
        array of records containing fields `time` and `evtid`

    Returns
    -------
    sorted awkward array
    """
    indices = np.lexsort((obj.time, obj.evtid))
    return obj[indices]


def group_by_evtid(data: Table) -> Table:
    """Simple grouping by evtid.

    Takes the input `stp` :class:`LGOD.Table` from remage and defines groupings of steps (i.e the
    `cumulative_length` for a vector of vectors). This then defines the output table (also :class:`LGDO.Table`),
    on which processors can add fields.

    Parameters
    ----------
    data
        LGDO Table which must contain the `evtid` field.

    Returns
    -------
    LGDO table of :class:`VectorOfVector` for each field.

    Note
    ----
    The input table must be sorted (by `evtid`).
    """

    # convert to awkward
    obj_ak = data.view_as("ak")

    # sort input
    obj_ak = sort_data(obj_ak)

    # extract cumulative lengths
    counts = ak.run_lengths(obj_ak.evtid)
    cumulative_length = np.cumsum(counts)

    # build output table
    out_tbl = Table(size=len(cumulative_length))

    for f in obj_ak.fields:
        out_tbl.add_field(
            f, VectorOfVectors(cumulative_length=cumulative_length, flattened_data=obj_ak[f])
        )
    return out_tbl


def group_by_time(data: Table, window: float = 10) -> lgdo.Table:
    """Grouping of steps by `evtid` and `time`.

    Takes the input `stp` :class:`LGOD.Table` from remage and defines groupings of steps (i.e the
    `cumulative_length` for a vector of vectors). This then defines the output table (also :class:`LGDO.Table`),
    on which processors can add fields.

    The windowing is based on defining a new group when the `evtid` changes or when the time increases by `> window`,
    which is in units of us.


    Parameters
    ----------
    data
        LGDO Table which must contain the `evtid`, `time` field.

    Returns
    -------
    LGDO table of :class:`VectorOfVector` for each field.

    Note
    ----
    The input table must be sorted (first by `evtid` then `time`).
    """

    obj = data.view_as("ak")
    obj = sort_data(obj)

    # get difference
    time_diffs = np.diff(obj.time)
    index_diffs = np.diff(obj.evtid)

    # index of thhe last element in each run
    time_change = (time_diffs > window * 1000) & (index_diffs == 0)
    index_change = index_diffs > 0

    # cumulative length is just the index of changes plus 1
    cumulative_length = np.array(np.where(time_change | index_change))[0] + 1

    # add the las grouping
    cumulative_length = np.append(cumulative_length, len(obj.time))

    # build output table
    out_tbl = Table(size=len(cumulative_length))

    for f in obj.fields:
        out_tbl.add_field(
            f, VectorOfVectors(cumulative_length=cumulative_length, flattened_data=obj[f])
        )

    return out_tbl


def smear_energies(truth_energy: Array, reso: float = 2) -> Array:
    """Smearing of energies.

    Parameters
    ----------
    truth_energy
        Array of energies to be smeared.
    reso
        energy resolution (sigma).

    Returns
    -------
    New array after sampling from a Gaussian with mean :math:`energy_i` and sigma `reso` for every element of `truth_array`.

    """

    flat_energy = truth_energy
    rng = np.random.default_rng()

    return Array(rng.normal(loc=flat_energy, scale=np.ones_like(flat_energy) * reso))


def distance_to_surface(
    positions_x: VectorOfVectors,
    positions_y: VectorOfVectors,
    positions_z: VectorOfVectors,
    hpge: legendhpges.base.HPGe,
    det_pos: ArrayLike,
    surface_type: str | None = None,
) -> Array:
    """Computes the distance from each step to the detector surface.

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

    Returns
    -------
    VectorOfVectors with the same shape as `positions_x/y/z` of the distance to the surface.

    Note
    ----
    `positions_x/positions_y/positions_z` must all have the same shape.

    """

    # compute local positions
    pos = []
    sizes = []
    for idx, pos_tmp in enumerate([positions_x, positions_y, positions_z]):
        local_pos_tmp = ak.Array(pos_tmp) - det_pos[idx]
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
    surface_indices = np.where(hpge.surfaces == surface_type) if surface_type is not None else None

    # distance calc itself
    distances = hpge.distance_to_surface(local_positions, surface_indices=surface_indices)

    return VectorOfVectors(ak.unflatten(distances, size))
