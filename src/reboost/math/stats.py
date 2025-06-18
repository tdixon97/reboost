from __future__ import annotations

import logging
from typing import Callable

import awkward as ak
import numpy as np
from lgdo import Array
from numpy.typing import ArrayLike

log = logging.getLogger(__name__)


def apply_energy_resolution(
    energies: ak.Array, channels: ak.Array, tcm_tables: dict, reso_pars: dict, reso_func: Callable
):
    """Apply the energy resolution sampling to an array with many channels.

    Parameters
    ----------
    energies
        the energies to smear
    channels
        the channel index for each energy
    tcm_tables
        the mapping from indices to channel names.
    reso_pars
        the pars for each channel.
    reso_func
        the function to compute the resolution.
    """
    n_pars = len(reso_pars[next(iter(reso_pars))])

    pars_shaped = []

    for _ in range(n_pars):
        pars_shaped.append(np.zeros(len(ak.flatten(channels))))

    num = ak.num(channels, axis=-1)

    for key in tcm_tables:
        for i in range(n_pars):
            pars_shaped[i][ak.flatten(channels) == key] = reso_pars[tcm_tables[key].split("/")[-1]][
                i
            ]

    ch_reso = reso_func(energies, *pars_shaped)

    energies_flat_smear = gaussian_sample(ak.flatten(energies), ch_reso)

    return ak.unflatten(energies_flat_smear, num)


def gaussian_sample(mu: ArrayLike, sigma: ArrayLike | float, *, seed: int = 999) -> Array:
    r"""Generate samples from a gaussian.

    Based on:

    .. math::

        y_i \sim \mathcal{N}(\mu_i,\sigma_i)

    where $y_i$ is the output, $x_i$ the input (mu) and $\sigma$ is the standard
    deviation for each point.

    Parameters
    ----------
    mu
        the mean positions to sample from, should be a flat (ArrayLike) object.
    sigma
        the standard deviation for each input value, can also be a single float.
    seed
        the random seed.

    Returns
    -------
    sampled values.
    """
    # convert inputs

    if isinstance(mu, Array):
        mu = mu.view_as("np")
    elif isinstance(mu, ak.Array):
        mu = mu.to_numpy()
    elif not isinstance(mu, np.ndarray):
        mu = np.array(mu)

    # similar for sigma
    if isinstance(sigma, Array):
        sigma = sigma.view_as("np")
    elif isinstance(sigma, ak.Array):
        sigma = sigma.to_numpy()
    elif not isinstance(sigma, (float, int, np.ndarray)):
        sigma = np.array(sigma)

    rng = np.random.default_rng(seed=seed)  # Create a random number generator

    return Array(rng.normal(loc=mu, scale=sigma))
