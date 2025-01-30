from __future__ import annotations

import logging

import awkward as ak
import numpy as np
from lgdo import Array
from numpy.typing import ArrayLike

log = logging.getLogger(__name__)


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
    `LGDO.Array` of sampled values.
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
