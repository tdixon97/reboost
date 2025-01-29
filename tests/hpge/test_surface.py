from __future__ import annotations

import awkward as ak
import numpy as np
import pyg4ometry
import pytest
from legendhpges import make_hpge
from legendtestdata import LegendTestData
from lgdo import types
from lgdo.types import VectorOfVectors

from reboost.hpge.surface import distance_to_surface


@pytest.fixture(scope="session")
def test_data_configs():
    ldata = LegendTestData()
    ldata.checkout("5f9b368")
    return ldata.get_path("legend/metadata/hardware/detectors/germanium/diodes")


def test_distance_to_surface(test_data_configs):
    gedet = make_hpge(test_data_configs + "/V99000A.json", registry=pyg4ometry.geant4.Registry())
    dist = [100, 0, 0]

    pos = ak.Array(
        {
            "xloc": [[0, 100, 200], [100], [700, 500, 200]],
            "yloc": [[100, 0, 0], [200], [100, 300, 200]],
            "zloc": [[700, 10, 20], [100], [300, 100, 0]],
            "distance_to_surface": [[1, 1, 1], [10], [1, 1, 1]],
        }
    )

    # check just the shape
    assert ak.all(
        ak.num(
            distance_to_surface(
                pos.xloc, pos.yloc, pos.zloc, gedet, det_pos=dist, surface_type=None
            ),
            axis=1,
        )
        == [3, 1, 3]
    )

    # check it can be written
    dist_full = distance_to_surface(
        pos.xloc, pos.yloc, pos.zloc, gedet, det_pos=dist, surface_type=None
    )
    assert isinstance(dist_full, types.LGDO)

    # check skipping the calculation for points > 5 mm
    dist = distance_to_surface(
        VectorOfVectors(pos.xloc),
        VectorOfVectors(pos.yloc),
        VectorOfVectors(pos.zloc),
        gedet,
        det_pos=dist,
        surface_type=None,
        distances_precompute=VectorOfVectors(pos.distance_to_surface),
        precompute_cutoff=5,
    )

    assert isinstance(dist, types.LGDO)

    assert ak.all(dist[0] == dist_full[0])
    assert ak.all(dist[2] == dist_full[2])

    assert np.isnan(dist[1][0])
