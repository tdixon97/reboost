from __future__ import annotations

import logging

import awkward as ak
import numba
import numpy as np
import pint
import pyg4ometry
from lgdo import Array, VectorOfVectors
from numpy.typing import ArrayLike, NDArray

from .. import units
from ..units import ureg as u
from .utils import HPGeScalarRZField

log = logging.getLogger(__name__)


def r90(edep: ak.Array, xloc: ak.Array, yloc: ak.Array, zloc: ak.Array) -> Array:
    """R90 HPGe pulse shape heuristic.

    Parameters
    ----------
    edep
        array of energy.
    xloc
        array of x coordinate position.
    yloc
        array of y coordinate position.
    zloc
        array of z coordinate position.
    """
    tot_energy = ak.sum(edep, axis=-1, keepdims=True)

    def eweight_mean(field, energy):
        return ak.sum(energy * field, axis=-1, keepdims=True) / tot_energy

    # Compute distance of each edep to the weighted mean
    dist = np.sqrt(
        (xloc - eweight_mean(edep, xloc)) ** 2
        + (yloc - eweight_mean(edep, yloc)) ** 2
        + (zloc - eweight_mean(edep, zloc)) ** 2
    )

    # Sort distances and corresponding edep within each event
    sorted_indices = ak.argsort(dist, axis=-1)
    sorted_dist = dist[sorted_indices]
    sorted_edep = edep[sorted_indices]

    def _ak_cumsum(layout, **_kwargs):
        if layout.is_numpy:
            return ak.contents.NumpyArray(np.cumsum(layout.data))

        return None

    # Calculate the cumulative sum of energies for each event
    cumsum_edep = ak.transform(
        _ak_cumsum, sorted_edep
    )  # Implement cumulative sum over whole jagged array
    if len(edep) == 1:
        cumsum_edep_corrected = cumsum_edep
    else:
        cumsum_edep_corrected = (
            cumsum_edep[1:] - cumsum_edep[:-1, -1]
        )  # correct to get cumsum of each lower level array
        cumsum_edep_corrected = ak.concatenate(
            [
                cumsum_edep[:1],  # The first element of the original cumsum is correct
                cumsum_edep_corrected,
            ]
        )

    threshold = 0.9 * tot_energy
    r90_indices = ak.argmax(cumsum_edep_corrected >= threshold, axis=-1, keepdims=True)
    r90 = sorted_dist[r90_indices]

    return Array(ak.flatten(r90).to_numpy())


def drift_time(
    xloc: ArrayLike,
    yloc: ArrayLike,
    zloc: ArrayLike,
    dt_map: HPGeScalarRZField,
    coord_offset: pint.Quantity | pyg4ometry.gdml.Position = (0, 0, 0) * u.m,
) -> VectorOfVectors:
    """Calculates drift times for each step (cluster) in an HPGe detector.

    Parameters
    ----------
    xloc
        awkward array of x coordinate position.
    yloc
        awkward array of y coordinate position.
    zloc
        awkward array of z coordinate position.
    dt_map
        the drift time map.
    coord_offset
        this `(x, y, z)` coordinates will be subtracted to (xloc, yloc, zloc)`
        before drift time computation. The length units must be the same as
        `xloc`, `yloc` and `zloc`.
    """
    # sanitize coord_offset
    coord_offset = units.pg4_to_pint(coord_offset)

    # unit handling (for matching with drift time map units)
    xu, yu = [units.units_convfact(data, dt_map.r_units) for data in (xloc, yloc)]
    zu = units.units_convfact(zloc, dt_map.z_units)

    # unwrap LGDOs
    xloc, yloc, zloc = [units.unwrap_lgdo(data)[0] for data in (xloc, yloc, zloc)]

    # awkward transform to apply the drift time map to the step coordinates
    def _ak_dt_map(layouts, **_kwargs):
        if layouts[0].is_numpy and layouts[1].is_numpy:
            return ak.contents.NumpyArray(
                dt_map.φ(np.stack([layouts[0].data, layouts[1].data], axis=1))
            )

        return None

    # transform coordinates
    xloc = xu * xloc - coord_offset[0].to(dt_map.r_units).m
    yloc = yu * yloc - coord_offset[1].to(dt_map.r_units).m
    zloc = zu * zloc - coord_offset[2].to(dt_map.z_units).m

    # evaluate the drift time
    dt_values = ak.transform(
        _ak_dt_map,
        np.sqrt(xloc**2 + yloc**2),
        zloc,
    )

    return VectorOfVectors(
        dt_values,
        attrs={"units": units.unit_to_lh5_attr(dt_map.φ_units)},
    )


def drift_time_heuristic(
    drift_time: ArrayLike,
    edep: ArrayLike,
) -> Array:
    """HPGe drift-time-based pulse-shape heuristic.

    See :func:`_drift_time_heuristic_impl` for a description of the algorithm.

    Parameters
    ----------
    drift_time
        drift time of charges originating from steps/clusters. Can be
        calculated with :func:`drift_time`.
    edep
        energy deposited in step/cluster (same shape as `drift_time`).
    """
    # extract LGDO data and units
    drift_time, t_units = units.unwrap_lgdo(drift_time)
    edep, e_units = units.unwrap_lgdo(edep)

    # we want to attach the right units to the dt heuristic, if possible
    attrs = {}
    if t_units is not None and e_units is not None:
        attrs["units"] = units.unit_to_lh5_attr(t_units / e_units)

    return Array(_drift_time_heuristic_impl(drift_time, edep), attrs=attrs)


@numba.njit(cache=True)
def _drift_time_heuristic_impl(
    dt: ak.Array,
    edep: ak.Array,
) -> NDArray:
    r"""Low-level implementation of the HPGe drift-time-based pulse-shape heuristic.

    Accepts Awkward arrays and uses Numba to speed up the computation.

    For each hit (collection of steps), the drift times and corresponding
    energies are sorted in ascending order. The function finds the optimal
    split point :math:`m` that maximizes the *identification metric*:

    .. math::

       I = \frac{|T_1 - T_2|}{E_\text{s}(E_1, E_2)}

    where:

    .. math::

        T_1 = \frac{\sum_{i < m} t_i E_i}{\sum_{i < m} E_i}
        \quad \text{and} \quad
        T_2 = \frac{\sum_{i \geq m} t_i E_i}{\sum_{i \geq m} E_i}

    are the energy-weighted mean drift times of the two groups.

    .. math::

        E_\text{scale}(E_1, E_2) = \frac{1}{\sqrt{E_1 E_2}}

    is the scaling factor.

    The function iterates over all possible values of :math:`m` and selects the
    maximum `I` as the drift time heuristic value.
    """
    dt_heu = np.zeros(len(dt))

    # loop over hits
    for i in range(len(dt)):
        t = np.asarray(dt[i])
        e = np.asarray(edep[i])

        valid_idx = np.where(e > 0)[0]
        if len(valid_idx) < 2:
            continue

        t = t[valid_idx]
        e = e[valid_idx]

        sort_idx = np.argsort(t)
        t = t[sort_idx]
        e = e[sort_idx]

        max_id_metric = 0
        for j in range(1, len(t)):
            e1 = np.sum(e[:j])
            e2 = np.sum(e[j:])

            t1 = np.sum(t[:j] * e[:j]) / e1
            t2 = np.sum(t[j:] * e[j:]) / e2

            id_metric = abs(t1 - t2) * np.sqrt(e1 * e2)

            max_id_metric = max(max_id_metric, id_metric)

        dt_heu[i] = max_id_metric

    return dt_heu
