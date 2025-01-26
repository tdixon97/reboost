from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from lgdo import VectorOfVectors

from reboost.math import functions


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
