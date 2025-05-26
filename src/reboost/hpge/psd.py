from __future__ import annotations

import logging
from math import erf, exp

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


@numba.njit(cache=True)
def _vectorized_erf(x: ArrayLike) -> NDArray:
    """Error function that can take in a numpy array."""
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = erf(x[i])
    return out


@numba.njit(cache=True)
def _current_pulse_model(
    times: ArrayLike, Amax: float, mu: float, sigma: float, tail_fraction: float, tau: float
) -> NDArray:
    r"""Analytic model for the current pulse in a Germanium detector.

    Consists of a Gaussian and an exponential tail:

     .. math::

       A(t) = A_{max}\times (1-p)\times \text{Gauss}(t,\mu,\sigma)+ A \times p (1-\text{Erf}((t-\mu)/sigma))\times
        \frac{e^{(t/\tau)}}{2e^{\mu/\tau}}

    Parameters
    ----------
    times
        Array of times to compute current for
    Amax
        Maximum current
    mu
        Time of the maximum current.
    sigma
        Width of the current pulse
    tail_fraction
        Fraction of the tail in the pulse.
    tau
        Time constant of the low time tail.

    Returns
    -------
    The predicted current waveform for this energy deposit.
    """
    norm = 2 * exp(mu / tau)

    dx = times - mu
    term1 = Amax * (1 - tail_fraction) * np.exp(-(dx * dx) / (2 * sigma * sigma))
    term2 = Amax * tail_fraction * (1 - _vectorized_erf(dx / sigma)) * np.exp(times / tau) / norm

    return term1 + term2


def convolve_surface_response(surf_current: np.ndarray, bulk_pulse: np.ndarray) -> NDArray:
    """Convolve the surface response pulse with the bulk current pulse.

    This combines the current induced on the edge of the FCCD region with the bulk response
    on the p+ contact.

    Parameters
    ----------
    surf_current
        array of the current induced via diffusion against time.
    bulk_pulse
        the pulse template to convolve the surface current with.

    Returns
    -------
    the current waveform after convolution.
    """
    return np.convolve(surf_current, bulk_pulse, mode="full")[: len(surf_current)]


@numba.njit(cache=True)
def get_current_waveform(
    edep: ak.Array,
    drift_time: ak.Array,
    sigma: float,
    tail_fraction: float,
    tau: float,
    norm: float,
    low: float,
    high: float,
    steps: float,
) -> tuple(NDArray, NDArray):
    r"""Estimate the current waveform.

    Based on modelling the current as a sum over the current pulse model in
    :func:`_current_pulse_model`.

    .. math::
        A(t) = \sum_i E_i \times N f(t,dt_i,\vector{\theta})

    Where:
        - $f(t)$ is the template
        - $\vector{\theta}$ are the parameters (sigma, p, tau)
        - $E_i$ and $dt_i$ are the deposited energy and drift time.
        - N is a normalisation term

    Parameters
    ----------
    edep
        Array of energies for each step
    drift_time
        Array of drift times for each step
    sigma
        Sigma parameter of the current pulse model.
    tail_fraction
        Tail-fraction parameter of the current pulse.
    tau
        Tail parameter of the current pulse
    norm
        Normalisation factor (from energy to current).
    low
        starting time for the waveform
    high
        ending time for the waveform
    steps
        number of steps

    Returns
    -------
    A tuple of the time and current for the current waveform for this event.
    """
    times = np.linspace(low, high, steps)
    y = np.zeros_like(times)

    for i in range(len(edep)):
        A = edep[i] * norm
        mu = drift_time[i]
        y += _current_pulse_model(
            times, A, mu=mu, sigma=sigma, tail_fraction=tail_fraction, tau=tau
        )

    return times, y


@numba.njit(cache=True)
def _estimate_current_impl(
    edep: ak.Array,
    dt: ak.Array,
    sigma: float,
    tail_fraction: float,
    tau: float,
    mean_AoE: float = 0,
) -> NDArray:
    """Estimate the maximum current that would be measured in the HPGe detector.

    This is based on extracting a waveform with :func:`get_current_waveform` and finding the maxima of it.
    Two iterations are performed first with ~ 10 ns time steps then a fine scan with 1 ns.

    Parameters
    ----------
    edep
        Array of energies for each step.
    drift_time
        Array of drift times for each step.
    sigma
        Sigma parameter of the current pulse model.
    tail_fraction
        Tail-fraction parameter of the current pulse.
    tau
        Tail parameter of the current pulse
    mean_AoE
        The mean AoE value for this detector (to normalise current pulses).
    """
    A = np.zeros(len(dt))

    # get normalisation factor
    x = np.linspace(3000, 0, 3000)
    y = _current_pulse_model(x, 1, 500, sigma, tail_fraction, tau)
    factor = np.max(y)

    for i in range(len(dt)):
        t = np.asarray(dt[i])
        e = np.asarray(edep[i])

        # first pass
        low, high = min(t), max(t)
        x, W = get_current_waveform(
            e,
            t,
            tau=tau,
            tail_fraction=tail_fraction,
            norm=mean_AoE / factor,
            sigma=sigma,
            low=low - 50,
            high=high + 50,
            steps=round((100 + high - low) / 10),
        )

        # second pass
        max_t = x[np.argmax(W)]

        x, W = get_current_waveform(
            e,
            t,
            tau=tau,
            tail_fraction=tail_fraction,
            norm=mean_AoE / factor,
            sigma=sigma,
            low=max_t - 20,
            high=max_t + 20,
            steps=40,
        )

        # save current
        A[i] = np.max(W)

    return A


def maximum_current(
    edep: ArrayLike,
    drift_time: ArrayLike,
    *,
    sigma: float,
    tail_fraction: float,
    tau: float,
    mean_AoE: float = 0,
) -> Array:
    """Estimate the maximum current in the HPGe detector based on :func:`_estimate_current_impl`.

    Parameters
    ----------
    edep
        Array of energies for each step.
    drift_time
        Array of drift times for each step.
    sigma
        Sigma parameter of the current pulse model.
    tail_fraction
        Tail-fraction parameter of the current pulse.
    tau
        Tail parameter of the current pulse
    mean_AoE
        The mean AoE value for this detector (to normalise current pulses).

    Returns
    -------
    An Array of the maximum current for each hit.
    """
    # extract LGDO data and units
    drift_time, time_unit = units.unwrap_lgdo(drift_time)

    if time_unit not in {"ns", "nanosecond", None}:
        msg = "Time unit must be ns"
        raise ValueError(msg)

    edep, _ = units.unwrap_lgdo(edep)

    return Array(
        _estimate_current_impl(
            ak.Array(edep),
            ak.Array(drift_time),
            sigma=sigma,
            tail_fraction=tail_fraction,
            tau=tau,
            mean_AoE=mean_AoE,
        )
    )
