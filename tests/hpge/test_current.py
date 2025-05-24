from __future__ import annotations

import awkward as ak
from lgdo import Array, VectorOfVectors

from reboost.hpge import psd


def test_maximum_current():
    edep = VectorOfVectors(ak.Array([[100, 300, 50], [10, 0, 100], [500]]), attrs={"unit": "keV"})
    times = VectorOfVectors(
        ak.Array([[400, 500, 700], [800, 0, 1500], [700]], attrs={"unit": "ns"})
    )

    curr = psd.maximum_current(edep, times, sigma=100, tau=100, tail_fraction=0.65, mean_AoE=0.5)
    assert isinstance(curr, Array)

    assert len(curr) == 3

    # should be close to 250 (could be some differences due to the discretisation)
    assert abs(curr[2] - 250) < 0.1
