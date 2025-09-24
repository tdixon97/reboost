from __future__ import annotations

import awkward as ak
import numpy as np
import pyg4ometry
import pytest
from legendhpges import make_hpge
from legendtestdata import LegendTestData

from reboost import units
from reboost.hpge.surface import distance_to_surface, get_surface_response
from reboost.units import ureg as u


@pytest.fixture(scope="session")
def test_data_configs():
    ldata = LegendTestData()
    ldata.checkout("5f9b368")
    return ldata.get_path("legend/metadata/hardware/detectors/germanium/diodes")


def test_distance_to_surface(test_data_configs):
    gedet = make_hpge(test_data_configs + "/V99000A.json", registry=pyg4ometry.geant4.Registry())
    dist = [100, 0, 0] * u.mm

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
    assert isinstance(dist_full, ak.Array)

    # check skipping the calculation for points > 5 mm
    dist = distance_to_surface(
        pos.xloc,
        pos.yloc,
        pos.zloc,
        gedet,
        det_pos=dist,
        surface_type=None,
        distances_precompute=pos.distance_to_surface,
        precompute_cutoff=5,
    )

    assert isinstance(dist, ak.Array)

    assert ak.all(dist[0] == dist_full[0])
    assert ak.all(dist[2] == dist_full[2])

    assert np.isnan(dist[1][0])


def test_units(test_data_configs):
    gedet = make_hpge(test_data_configs + "/V99000A.json", registry=pyg4ometry.geant4.Registry())
    dist = [100, 0, 0] * u.mm

    pos = ak.Array(
        {
            "xloc": units.attach_units([[0, 100, 200], [100], [700, 500, 200]], "mm"),
            "yloc": units.attach_units([[100, 0, 0], [200], [100, 300, 200]], "mm"),
            "zloc": units.attach_units([[700, 10, 20], [100], [300, 100, 0]], "mm"),
            "distance_to_surface": units.attach_units([[1, 1, 1], [10], [1, 1, 1]], "m"),
        }
    )

    dists = distance_to_surface(
        pos.xloc, pos.yloc, pos.zloc, gedet, det_pos=dist, surface_type=None
    )

    assert units.get_unit_str(dists) == "mm"

    pos_m = ak.Array(
        {
            "xloc": units.attach_units([[0, 0.100, 0.200], [0.100], [0.700, 0.500, 0.200]], "m"),
            "yloc": units.attach_units([[0.100, 0, 0], [0.200], [0.100, 0.300, 0.200]], "m"),
            "zloc": units.attach_units([[0.700, 0.01, 0.02], [0.100], [0.300, 0.100, 0]], "m"),
            "distance_to_surface": units.attach_units([[1, 1, 1], [10], [1, 1, 1]], "m"),
        }
    )
    dist_m = distance_to_surface(
        pos_m.xloc, pos_m.yloc, pos_m.zloc, gedet, det_pos=dist, surface_type=None
    )

    assert units.get_unit_str(dist_m) == "mm"

    assert ak.all(dists == dist_m)


def test_surface_response():
    response = get_surface_response(fccd=1000, init=500)

    # should be 10000 samples
    assert len(response) == 10000
