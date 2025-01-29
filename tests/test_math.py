from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from lgdo import Array, VectorOfVectors

from reboost.math import functions, stats


def test_hpge_activeness():
    # test with VectorOfVectors and ak.Array input
    distances = ak.Array([[0.2], [0.6], [2]])

    activeness = functions.piecewise_linear_activeness(VectorOfVectors(distances), fccd=1, tl=0.5)

    # first point should be 0
    assert activeness[0][0] == 0
    # second should be 0.1/0.5 = 0.2
    assert activeness[1][0] == pytest.approx(0.2)
    assert activeness[2][0] == 1

    # test with ak.Array input
    distances = ak.Array([[0.2], [0.6], [2]])
    activeness = functions.piecewise_linear_activeness(distances, fccd=1, tl=0.5)

    # first point should be 0
    assert activeness[0][0] == 0
    # second should be 0.1/0.5 = 0.2
    assert activeness[1][0] == pytest.approx(0.2)
    assert activeness[2][0] == 1

    # test with Array
    activeness = functions.piecewise_linear_activeness([0.2, 0.6, 2], fccd=1, tl=0.5)
    assert np.allclose(activeness.view_as("np"), [0, 0.2, 1])


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
