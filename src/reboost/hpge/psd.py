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
def _njit_erf(x: ArrayLike) -> NDArray:
    """Error function that can take in a numpy array."""
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = erf(x[i])
    return out


@numba.njit(cache=True)
def _current_pulse_model(
    times: ArrayLike,
    amax: float,
    mu: float,
    sigma: float,
    tail_fraction: float,
    tau: float,
    high_tail_fraction: float = 0,
    high_tau: float = 0,
) -> NDArray:
    r"""Analytic model for the current pulse in a Germanium detector.

    Consists of a Gaussian, a high side exponential tail and a low side tail:

     .. math::

       A(t) = A_{max}\times (1-p-p_h)\times \text{Gauss}(t,\mu,\sigma)+ A \times p (1-\text{Erf}((t-\mu)/sigma_i))\times
        \frac{e^{( t/\tau)}}{2e^{\mu/\tau}} + A \times p_h (1-\text{Erf}(-(t-\mu)/sigma_i))\times
        \frac{e^{-( t/\tau)}}{2}

    Parameters
    ----------
    times
        Array of times to compute current for.
    amax
        Maximum current for the template
    mu
        Time of the maximum current.
    sigma
        Width of the current pulse
    tail_fraction
        Fraction of the tail in the pulse.
    tau
        Time constant of the low time tail.
    high__tail_fraction
        Fraction of the high tail in the pulse.
    high_tau
        Time constant of the high time tail.

    Returns
    -------
    The predicted current waveform for this energy deposit.
    """
    norm = 2 * exp(mu / tau)
    norm_high = 2

    dx = times - mu
    term1 = (
        amax * (1 - tail_fraction - high_tail_fraction) * np.exp(-(dx * dx) / (2 * sigma * sigma))
    )
    term2 = amax * tail_fraction * (1 - _njit_erf(dx / sigma)) * np.exp(times / tau) / norm
    term3 = (
        amax
        * high_tail_fraction
        * (1 - _njit_erf(-dx / sigma))
        * np.exp(-(times - mu) / high_tau)
        / norm_high
    )

    return term1 + term2 + term3


@numba.njit(cache=True)
def _interpolate_pulse_model(
    template: Array, time: float, start: float, end: float, dt: float, mu: float
) -> NDArray:
    """Interpolate to extract the pulse model given a particular mu."""
    local_time = time - mu - start

    if (local_time < start) or (int(local_time) > end):
        return 0

    sample = int(local_time / dt)
    A_before = template[sample]
    A_after = template[sample + 1]

    frac = (local_time - int(local_time)) / dt
    return A_before + frac * (A_after - A_before)


def make_convolved_surface_library(bulk_template: np.array, surface_library: np.array) -> NDArray:
    """Make the convolved surface library out of the template.

    This convolves every row of the surface_library with the template and reshapes the output
    to match the initial template. It returns a 2D array with one more row than the surface_library
    and each row the same length as the template. The final row is the bulk_template for easier interpolation.

    Parameters
    ----------
    bulk_template
        The template for the bulk response
    surface_library
        The 2D array of the surface library.

    Returns
    -------
    2D array of the surface library convolved with the bulk response.
    """
    # force surface library to be 2D
    if surface_library.ndim == 1:
        surface_library = surface_library.reshape((-1, 1))

    templates = np.zeros((len(bulk_template), np.shape(surface_library)[1] + 1))

    for i in range(np.shape(surface_library)[1]):
        templates[:, i] = convolve_surface_response(
            surface_library[1:, i] - surface_library[:-1, i], bulk_template
        )[: len(bulk_template)]

    templates[:, -1] = bulk_template

    return templates


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
    template: ArrayLike,
    start: float,
    dt: float,
    range_t: tuple,
) -> tuple(NDArray, NDArray):
    r"""Estimate the current waveform.

    Based on modelling the current as a sum over the current pulse model defined by
    the template.

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
    template
        array of the template for the current waveforms
    start
        first time value of the template
    dt
        timestep (in ns) for the template.
    range_t
        a range of times to search around

    Returns
    -------
    A tuple of the time and current for the current waveform for this event.
    """
    n = len(template)

    times = np.arange(n) * dt + start
    y = np.zeros_like(times, dtype=np.float64)

    for j in range(n):
        time = start + dt * j
        if (time < range_t[0]) or (time > (range_t[1] - dt)):
            continue
        y[j] = _get_waveform_value(j, edep, drift_time, template, start, dt, range_t)

    return times, y


@numba.njit(cache=True)
def _get_waveform_value_surface(
    idx: int,
    edep: NDArray,
    drift_time: np.array,
    dist_to_nplus: np.array,
    bulk_template: ArrayLike,
    templates_surface: ArrayLike,
    activeness_surface: ArrayLike,
    distance_step_in_um: float,
    fccd: float,
    start: float,
    dt: float,
) -> tuple[float, float]:
    """Get the value of the waveform at a certain index.

    Parameters
    ----------
    idx
        the index of the time array to find the waveform at.
    edep
        Array of energies for each step
    drift_time
        Array of drift times for each step
    template
        array of the template for the current waveforms
    templates_surface
        The current templates from the surface.
    activeness_surface
        The total collected charge for each surface point.
    dist_step_in_um
        The binning in distance for the surface pulse library.
    start
        first time value of the template
    dt
        timestep (in ns) for the template.

    Returns
    -------
    Value of the current waveform and the energy.
    """
    n = len(bulk_template)
    out = 0
    etmp = 0
    time = start + dt * idx

    for i in range(len(edep)):
        E = edep[i]
        mu = drift_time[i]
        dist = dist_to_nplus[i]

        if dist < fccd:
            dist_bin = int(dist / distance_step_in_um)

            # get two values (to allow linear interpolation)
            value_low = _interpolate_pulse_model(
                templates_surface[dist_bin], time, start, start + dt * n, dt, mu
            )
            value_high = _interpolate_pulse_model(
                templates_surface[dist_bin + 1], time, start, start + dt * n, dt, mu
            )

            # interpolate between distance bins
            diff = dist / distance_step_in_um - dist_bin
            out += E * (value_low + diff * (value_high - value_low))

            act_low = activeness_surface[dist_bin]
            act_high = activeness_surface[dist_bin + 1]
            etmp += (act_low + diff * (act_high - act_low)) * E

        else:
            out += E * _interpolate_pulse_model(bulk_template, time, start, start + dt * n, dt, mu)
            etmp += E
    return out, etmp


@numba.njit(cache=True)
def _get_waveform_value(
    idx: int,
    edep: ak.Array,
    drift_time: ak.Array,
    template: ArrayLike,
    start: float,
    dt: float,
) -> float:
    """Get the value of the waveform at a certain index.

    Parameters
    ----------
    idx
        the index of the time array to find the waveform at.
    edep
        Array of energies for each step
    drift_time
        Array of drift times for each step
    template
        array of the template for the current waveforms
    start
        first time value of the template
    dt
        timestep (in ns) for the template.

    Returns
    -------
    Value of the current waveform
    """
    n = len(template)
    out = 0
    time = start + dt * idx

    for i in range(len(edep)):
        E = edep[i]
        mu = drift_time[i]

        out += E * _interpolate_pulse_model(template, time, start, start + dt * n, dt, mu)

    return out


def get_current_template(
    low: float = -1000, high: float = 4000, step: float = 1, mean_aoe: float = 1, **kwargs
) -> tuple[NDArray, NDArray]:
    """Build the current template from the analytic model, defined by :func:`_current_pulse_model`.

    Parameters
    ----------
    low
        start of the template
    high
        end of the template
    step
        time-step, this should divide high-low
    mean_aoe
        The mean AoE value for this detector (to normalise current pulses).
    **kwargs
        Other keyword arguments passed to :func:`_current_pulse_model`.

    Returns
    -------
    tuple of the (template,times)
    """
    if int((high - low) / step) != (high - low) / step:
        msg = "Time template is not a multiple of the time-step."
        raise ValueError(msg)

    x = np.linspace(low, high, int((high - low) / step) + 1)
    template = _current_pulse_model(x, **kwargs)
    template /= np.max(template)
    template *= mean_aoe

    return template, x


@numba.njit(cache=True)
def _get_waveform_maximum_impl(
    t: ArrayLike,
    e: ArrayLike,
    dist: ArrayLike,
    template: ArrayLike,
    templates_surface: ArrayLike,
    activeness_surface: ArrayLike,
    tmin: float,
    tmax: float,
    start: float,
    fccd: float,
    n: int,
    time_step: int,
    surface_step_in_um: float,
    include_surface_effects: bool,
):
    """Basic implementation to get the maximum of the waveform.

    Parameters
    ----------
    t
        drift time for each step.
    e
        energy for each step.
    dist
        distance to surface for each step.
    """
    max_a = 0
    max_t = 0
    energy = np.sum(e)

    for j in range(0, n, time_step):
        time = start + j

        # skip anything not in the range tmin to tmax (for surface affects this can be later)
        has_surface_hit = include_surface_effects

        if time < tmin or (time > (tmax + time_step)):
            continue

        if not has_surface_hit:
            val_tmp = _get_waveform_value(j, e, t, template, start=start, dt=1.0)
        else:
            val_tmp, energy = _get_waveform_value_surface(
                j,
                e,
                t,
                dist,
                template,
                templates_surface.T,
                activeness_surface,
                distance_step_in_um=surface_step_in_um,
                fccd=fccd,
                start=start,
                dt=1.0,
            )

        if val_tmp > max_a:
            max_t = time
            max_a = val_tmp

    return max_t, max_a, energy


@numba.njit(cache=True)
def _estimate_current_impl(
    edep: ak.Array,
    dt: ak.Array,
    dist_to_nplus: ak.Array,
    template: np.array,
    times: np.array,
    include_surface_effects: bool,
    fccd: float,
    templates_surface: np.array,
    activeness_surface: np.array,
    surface_step_in_um: float,
) -> tuple[NDArray, NDArray, NDArray]:
    """Estimate the maximum current that would be measured in the HPGe detector.

    This is based on extracting a waveform with :func:`get_current_waveform` and finding the maxima of it.

    Parameters
    ----------
    edep
        Array of energies for each step.
    dt
        Array of drift times for each step.
    dist_to_nplus
        Array of distance to nplus contact for each step (can be `None`, in which case no surface effects are included.)
    template
        array of the bulk pulse template
    times
        time-stamps for the bulk pulse template
    """
    A = np.zeros(len(dt))
    maximum_t = np.zeros(len(dt))
    energy = np.zeros(len(dt))

    time_step = 1
    n = len(template)
    start = times[0]

    if include_surface_effects:
        offsets = times[np.argmax(templates_surface, axis=0)]

    # make the convolved surface library
    if include_surface_effects and np.diff(times)[0] != 1.0:
        msg = "The surface convolution requires a template with 1 ns binning"
        raise ValueError(msg)

    for i in range(len(dt)):
        t = np.asarray(dt[i])
        e = np.asarray(edep[i])
        dist = np.asarray(dist_to_nplus[i])

        # get the expected maximum
        tmax = float(np.max(t))
        tmin = float(np.min(t))

        # correct the maximum expected time for surface sims
        if include_surface_effects:
            ncols = templates_surface.shape[1]

            for j, d in enumerate(dist):
                dtmp = int(d / surface_step_in_um)

                # Use branchless selection
                use_offset = dtmp <= ncols
                offset_val = offsets[dtmp] if use_offset else 0.0
                time_tmp = t[j] + offset_val * use_offset

                tmax = max(tmax, time_tmp)

        for time_step in [20, 1]:
            if time_step == 1:
                tmin = int(maximum_t[i] - 50)
                tmax = int(maximum_t[i] + 50)

            # get the value
            maximum_t[i], A[i], energy[i] = _get_waveform_maximum_impl(
                t,
                e,
                dist,
                template,
                templates_surface,
                activeness_surface,
                tmin=tmin,
                tmax=tmax,
                start=start,
                fccd=fccd,
                n=n,
                time_step=time_step,
                surface_step_in_um=surface_step_in_um,
                include_surface_effects=include_surface_effects,
            )

    return A, maximum_t, energy


def maximum_current(
    edep: ArrayLike,
    drift_time: ArrayLike,
    dist_to_nplus: ArrayLike | None = None,
    *,
    template: np.array,
    times: np.array,
    fccd_in_um: float = 0,
    templates_surface: ArrayLike | None = None,
    activeness_surface: ArrayLike | None = None,
    surface_step_in_um: float = 10,
    return_mode: str = "current",
) -> Array:
    """Estimate the maximum current in the HPGe detector based on :func:`_estimate_current_impl`.

    Parameters
    ----------
    edep
        Array of energies for each step.
    drift_time
        Array of drift times for each step.
    dist_to_nplus
        Distance to n-plus electrode, only needed if surface heuristics are enabled.
    template
        array of the bulk pulse template
    times
        time-stamps for the bulk pulse template
    fccd
        Value of the full-charge-collection depth, if `None` no surface corrections are performed.
    surface_library
        2D array (distance, time) of the rate of charge arriving at the p-n junction. Each row
        should be an array of length 10000 giving the charge arriving at the p-n junction for each timestep
        (in ns). This is produced by :func:`.hpge.surface.get_surface_response` or other libraries.
    surface_step_in_um
        Distance step for the surface library.
    return_mode
        either current, energy or max_time

    Returns
    -------
    An Array of the maximum current/ time / energy for each hit.
    """
    # extract LGDO data and units

    drift_time, _ = units.unwrap_lgdo(drift_time)
    edep, _ = units.unwrap_lgdo(edep)
    dist_to_nplus, _ = units.unwrap_lgdo(dist_to_nplus)

    include_surface_effects = False

    if templates_surface is not None:
        if dist_to_nplus is None:
            msg = "Surface effects requested but distance not provided"
            raise ValueError(msg)

        include_surface_effects = True
    else:
        # convert types to keep numba happy
        templates_surface = np.zeros((1, len(template)))
        dist_to_nplus = ak.full_like(edep, np.nan)

    # convert types for numba
    if activeness_surface is None:
        activeness_surface = np.zeros(len(template))

    if not ak.all(ak.num(edep, axis=-1) == ak.num(drift_time, axis=-1)):
        msg = "edep and drift time must have the same shape"
        raise ValueError(msg)

    curr, time, energy = _estimate_current_impl(
        ak.values_astype(ak.Array(edep), np.float64),
        ak.values_astype(ak.Array(drift_time), np.float64),
        ak.values_astype(ak.Array(dist_to_nplus), np.float64),
        template=template,
        times=times,
        fccd=fccd_in_um,
        include_surface_effects=include_surface_effects,
        templates_surface=templates_surface,
        activeness_surface=activeness_surface,
        surface_step_in_um=surface_step_in_um,
    )

    # return
    if return_mode == "max_time":
        return Array(time)
    if return_mode == "current":
        return Array(curr)
    if return_mode == "energy":
        return Array(energy)

    msg = f"Return mode {return_mode} is not implemented."
    raise ValueError(msg)
