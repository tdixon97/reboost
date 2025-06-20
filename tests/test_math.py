from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from lgdo import Array, VectorOfVectors

from reboost.math import functions, stats


def test_hpge_activeness():
    # test with VectorOfVectors and ak.Array input
    distances = ak.Array([[0.2], [0.6], [2]])

    activeness = functions.piecewise_linear_activeness(VectorOfVectors(distances), fccd=1, dlf=0.4)

    # first point should be 0
    assert activeness[0][0] == 0
    # second should be 0.1/0.5 = 0.2
    assert activeness[1][0] == pytest.approx(1 / 3.0)
    assert activeness[2][0] == 1

    # test with ak.Array input
    distances = ak.Array([[0.2], [0.6], [2]])
    activeness = functions.piecewise_linear_activeness(distances, fccd=1, dlf=0.4)

    # first point should be 0
    assert activeness[0][0] == 0
    # second should be 0.1/0.5 = 0.2
    assert activeness[1][0] == pytest.approx(1 / 3.0)
    assert activeness[2][0] == 1

    # test with Array
    activeness = functions.piecewise_linear_activeness([[0.2, 0.6, 2]], fccd=1, dlf=0.4)

    assert np.allclose(activeness.view_as("np"), [0, 1 / 3.0, 1])

    # test with array
    distances = [0.2, 0.6, 2]
    activeness = functions.piecewise_linear_activeness(Array(distances), fccd=1, dlf=0.4)

    # first point should be 0
    assert activeness[0] == 0
    # second should be 0.1/0.5 = 0.2
    assert activeness[1] == pytest.approx(1 / 3.0)
    assert activeness[2] == 1


def test_vectorised_activeness():
    # vary fccd
    distances = ak.Array([[0.2, 5], [0.6], [2], [5]])
    edep = ak.Array([[100, 200], [100], [100], [100]])

    # simple case
    energy = functions.vectorised_active_energy(
        VectorOfVectors(distances), VectorOfVectors(edep), fccd=1, dlf=0.4
    )

    assert energy[0] == 200
    assert energy[1] == pytest.approx(100 / 3.0)
    assert energy[2] == 100

    # now vectorised over fccd
    energy_fccd = functions.vectorised_active_energy(
        VectorOfVectors(distances), VectorOfVectors(edep), fccd=[0, 0.5, 1, 2, 3], dlf=1
    )

    # with distance of 0.2 only fccd of 0 gives energy
    assert np.allclose(energy_fccd[0], [300, 200, 200, 200, 200])
    assert np.allclose(energy_fccd[1], [100, 100, 0, 0, 0])
    assert np.allclose(energy_fccd[2], [100, 100, 100, 0, 0])
    assert np.allclose(energy_fccd[3], [100, 100, 100, 100, 100])

    # vectorise over the dlf
    energy_dlf = functions.vectorised_active_energy(
        VectorOfVectors(distances), VectorOfVectors(edep), fccd=1, dlf=[0.4, 1]
    )

    assert np.allclose(energy_dlf[0], [200, 200])
    assert np.allclose(energy_dlf[1], [100 / 3.0, 0])
    assert np.allclose(energy_dlf[2], [100, 100])
    assert np.allclose(energy_dlf[3], [100, 100])


def test_sample():
    # list inputs
    samples = stats.gaussian_sample([1, 2, 3], [0.1, 0.1, 0.1])
    assert isinstance(samples, Array)

    # LGDO inputs
    samples = stats.gaussian_sample(Array(np.array([1, 2, 3])), Array(np.array([0.1, 0.1, 0.1])))
    assert isinstance(samples, Array)

    # ak inputs
    samples = stats.gaussian_sample(ak.Array([1, 2, 3]), ak.Array([1, 2, 3]))
    assert isinstance(samples, Array)

    # sigma float
    samples = stats.gaussian_sample([1, 2, 3], 0.1)
    assert isinstance(samples, Array)


def test_energy_res():
    energy = ak.Array([[100, 100], [200], [300, 100, 100]])
    channels = ak.Array([[0, 1], [1], [2, 0, 1]])

    tcm_tables = {"det000": 0, "det001": 1, "det002": 2}

    reso_pars = {"det000": [1, 0], "det001": [1, 0.01], "det002": [2, 0.05]}

    def reso_func(energy, p0, p1):
        return np.sqrt(energy * p1 + p0)

    reso = stats.get_resolution(energy, channels, tcm_tables, reso_pars, reso_func)

    assert len(reso) == len(energy)
    assert ak.all(ak.num(reso, axis=-1) == ak.num(energy, axis=-1))

    # test a few values
    assert reso[0][0] == np.sqrt(100 * 0 + 1)
    assert reso[0][1] == np.sqrt(100 * 0.01 + 1)

    smeared = stats.apply_energy_resolution(energy, channels, tcm_tables, reso_pars, reso_func)
    assert len(smeared) == len(energy)
    assert ak.all(ak.num(smeared, axis=-1) == ak.num(energy, axis=-1))
