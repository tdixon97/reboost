from __future__ import annotations

import awkward as ak
import pytest
from legendhpges import make_hpge
from legendtestdata import LegendTestData

from reboost.hpge.processors import distance_to_surface


@pytest.fixture(scope="session")
def test_data_configs():
    ldata = LegendTestData()
    ldata.checkout("5f9b368")
    return ldata.get_path("legend/metadata/hardware/detectors/germanium/diodes")


def test_distance_to_surface(test_data_configs):
    gedet = make_hpge(test_data_configs + "/V99000A.json")
    dist = [100, 0, 0]

    pos = ak.Array(
        {
            "xloc": [[0, 100, 200], [100], [700, 500, 200]],
            "yloc": [[100, 0, 0], [200], [100, 300, 200]],
            "zloc": [[700, 10, 20], [100], [300, 100, 0]],
        }
    )

    # check just the shape
    assert ak.all(
        ak.num(distance_to_surface(pos.xloc, pos.yloc, pos.zloc, gedet, dist, None), axis=1)
        == [3, 1, 3]
    )
