from __future__ import annotations

import logging

import awkward as ak
import numba
import numpy as np
from lgdo import VectorOfVectors

from .. import units

log = logging.getLogger(__name__)


def apply_cluster(cluster_run_lengths: ak.Array, field: ak.Array) -> ak.Array:
    """Apply clustering to a field.

    Parameters
    ----------
    cluster_ids
        run lengths of each cluster
    field
        the field to cluster
    """
    if isinstance(cluster_run_lengths, VectorOfVectors):
        cluster_run_lengths = cluster_run_lengths.view_as("ak")

    if isinstance(field, VectorOfVectors):
        field = field.view_as("ak")

    n_cluster = ak.num(cluster_run_lengths, axis=-1)
    clusters = ak.unflatten(ak.flatten(field), ak.flatten(cluster_run_lengths))

    # reshape into cluster oriented
    return ak.Array(ak.unflatten(clusters, n_cluster), attrs=field.attrs)


def cluster_by_step_length(
    trackid: ak.Array,
    pos_x: ak.Array,
    pos_y: ak.Array,
    pos_z: ak.Array,
    dist: ak.Array | None = None,
    surf_cut: float | None = None,
    threshold_in_mm: float = 0.1,
    threshold_surf_in_mm: float | None = None,
) -> VectorOfVectors:
    """Perform clustering based on the step length.

    Steps are clustered based on distance, if either:
     - a step is in a new track,
     - a step moves from surface to bulk region (or visa versa),
     - the distance between the current step and the first step of the current cluster is above a threshold.

    Then a new cluster is started. The surface region is defined as the volume
    less than surf_cut distance to the surface. This allows for a fine tuning of the
    parameters to be different for bulk and surface.

    Parameters
    ----------
    trackid
        index of the tracks.
    pos_x
        x position of the steps.
    pos_y
        y position of the steps.
    pos_z
        z position of the steps.
    dist
        distance to the detector surface. Can be `None` in which case all steps are treated as being in the "bulk".
    surf_cut
        Size of the surface region (in mm), if `None` no selection is applied (default).
    threshold_in_mm
        Distance threshold in mm to combine steps in the bulk.
    threshold_surf_in_mm
        Distance threshold in mm to combine steps in the surface.

    Returns
    -------
    Array of the run lengths of each cluster within a hit.
    """
    # type conversions

    pos = np.vstack(
        [
            ak.flatten(units.units_conv_ak(p, "mm")).to_numpy().astype(np.float64)
            for p in [pos_x, pos_y, pos_z]
        ]
    ).T

    indices_flat = _cluster_by_distance_numba(
        ak.flatten(ak.local_index(trackid)).to_numpy(),
        ak.flatten(trackid).to_numpy(),
        pos,
        dist_to_surf=ak.flatten(dist).to_numpy() if dist is not None else dist,
        surf_cut=surf_cut,
        threshold=threshold_in_mm,
        threshold_surf=threshold_surf_in_mm,
    )

    # reshape into being event oriented
    indices = ak.unflatten(indices_flat, ak.num(ak.local_index(trackid)))

    # number of steps per cluster
    counts = ak.run_lengths(indices)

    return ak.Array(counts)


@numba.njit
def _cluster_by_distance_numba(
    local_index: np.ndarray,
    trackid: np.ndarray,
    pos: np.ndarray,
    dist_to_surf: np.ndarray | None,
    surf_cut: float | None = None,
    threshold: float = 0.1,
    threshold_surf: float | None = None,
) -> np.ndarray:
    """Cluster steps by the distance between points in the same track.

    This function gives the basic numerical calculations for
    :func:`cluster_by_step_length`.

    Parameters
    ----------
    local_index
        1D array of the local index within each hit (step group)
    trackid
        1D array of index of the track
    pos
        `(n,3)` size array of the positions
    dist_to_surf
        1D array of the distance to the detector surface. Can be `None` in which case all steps are treated as being in the bulk.
    surf_cut
        Size of the surface region (in mm), if `None` no selection is applied.
    threshold
        Distance threshold in mm to combine steps in the bulk.
    threshold_surf
        Distance threshold in mm to combine steps in the surface.

    Returns
    -------
    np.ndarray
        1D array of cluster indices
    """

    def _dist(a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    n = len(local_index)
    out = np.zeros((n,), dtype=numba.int32)

    trackid_prev = -1
    pos_prev = np.zeros(3, dtype=numba.float64)
    cluster_idx = -1
    is_surf_prev = False

    for idx in range(n):
        # consider a surface and a bulk region
        if dist_to_surf is not None:
            thr = threshold if dist_to_surf[idx] > surf_cut else threshold_surf

            new_cluster = (
                (trackid[idx] != trackid_prev)
                or (is_surf_prev and (dist_to_surf[idx] > surf_cut))
                or ((not is_surf_prev) and (dist_to_surf[idx] < surf_cut))
                or (_dist(pos[idx, :], pos_prev) > thr)
            )
        # basic clustering without split into surface / bulk
        else:
            thr = threshold
            new_cluster = (trackid[idx] != trackid_prev) or (_dist(pos[idx, :], pos_prev) > thr)

        # New hit, reset cluster index
        if idx == 0 or local_index[idx] == 0:
            cluster_idx = 0
            pos_prev = pos[idx]

        # either new track, moving from surface to bulk,
        # moving from bulk to surface, or stepping more than
        # the threshold. Start a new cluster.
        elif new_cluster:
            cluster_idx += 1
            pos_prev = pos[idx, :]

        out[idx] = cluster_idx

        # Update previous values
        trackid_prev = trackid[idx]
        if dist_to_surf is not None:
            is_surf_prev = dist_to_surf[idx] < surf_cut

    return out


def step_lengths(
    x_cluster: ak.Array,
    y_cluster: ak.Array,
    z_cluster: ak.Array,
) -> ak.Array:
    """Compute the distance between consecutive steps.

    This is based on calculating the distance between consecutive steps in the same track,
    thus the input arrays should already be clustered (have dimension 3). The output
    will have a similar shape to the input with one less entry in the outermost dimension.

    Example config (assuming that the clustered positions are obtained already):

    .. code-block:: yaml

        step_lengths: reboost.shape.cluster.step_lengths(HITS.cluster_x,HITS.cluster_y,HITS.cluster_z))

    Parameters
    ----------
    x_cluster
        The x location of each step in each cluster and event.
    y_cluster
        The y location of each step in each cluster and event.
    z_cluster
        The z location of each step in each cluster and event.

    Returns
    -------
    a `VectorOfVectors` of the step lengths in each cluster.
    """
    data = [x_cluster, y_cluster, z_cluster]

    for idx, var in enumerate(data):
        # check shape
        if var.ndim != 3:
            msg = f"The input array for step lengths must be 3 dimensional not {data[idx.dim]}"
            raise ValueError(msg)

        # type convert
        data[idx] = units.units_conv_ak(data[idx], "mm")

    counts = ak.num(data[0], axis=-1)
    data = np.vstack([ak.flatten(ak.flatten(var)).to_numpy() for var in data])
    dist = np.append(np.sqrt(np.sum(np.diff(data, axis=1) ** 2, axis=0)), 0)

    n_cluster = ak.num(counts, axis=-1)
    clusters = ak.unflatten(ak.Array(dist), ak.flatten(counts))

    out = ak.unflatten(clusters, n_cluster)
    return ak.Array(out[:, :, :-1], attrs={"units": "mm"})
