from __future__ import annotations

import pathlib

import awkward as ak
import numpy as np
import pyg4ometry
import pytest
from legendhpges import make_hpge
from legendtestdata import LegendTestData
from lgdo import Table
from pyg4ometry import geant4

from reboost.hpge import hit, utils

configs = pathlib.Path(__file__).parent.resolve() / pathlib.Path("configs")


@pytest.fixture(scope="session")
def test_data_configs():
    ldata = LegendTestData()
    ldata.checkout("5f9b368")
    return ldata.get_path("legend/metadata/hardware/detectors/germanium/diodes")


def test_eval():
    in_arr = ak.Array(
        {
            "evtid": [[1, 1, 1], [2, 2], [10, 10], [11], [12, 12, 12]],
            "time": [[0, 0, 0], [0, 0], [0, 0], [0], [0, 0, 0]],
            "edep": [[100, 150, 300], [0, 2000], [10, 20], [19], [100, 200, 300]],
        }
    )
    tab = Table(in_arr)
    basic_eval = {"mode": "eval", "expression": "ak.sum(hit.edep,axis=-1)"}

    assert ak.all(
        hit.eval_expression(tab, basic_eval, {}).view_as("ak") == [550, 2000, 30, 19, 600]
    )

    tab.add_field("e_sum", hit.eval_expression(tab, basic_eval, {}))
    bad_eval = {"mode": "sum", "expression": "ak.sum(hit.edep,axis=-1)"}

    with pytest.raises(ValueError):
        hit.eval_expression(tab, bad_eval, {})

    func_eval = {
        "mode": "function",
        "expression": "reboost.hpge.processors.smear_energies(hit.e_sum,reso=pars.reso)",
    }
    pars = {"reso": 2}

    assert np.size(hit.eval_expression(tab, func_eval, pars).view_as("np")) == 5


def test_eval_with_hpge(test_data_configs):
    reg = geant4.Registry()
    gedet = make_hpge(test_data_configs + "/V99000A.json", registry=reg)

    pos = ak.Array(
        {
            "xloc": [[0, 100, 200], [100], [700, 500, 200]],
            "yloc": [[100, 0, 0], [200], [100, 300, 200]],
            "zloc": [[700, 10, 20], [100], [300, 100, 0]],
        }
    )
    tab = Table(pos)

    func_eval = {
        "mode": "function",
        "expression": "reboost.hpge.processors.distance_to_surface(hit.xloc, hit.yloc, hit.zloc, hpge, [0,0,0], None)",
    }

    assert ak.all(
        ak.num(hit.eval_expression(tab, func_eval, {}, hpge=gedet, phy_vol=None), axis=1)
        == [3, 1, 3]
    )


def test_eval_with_hpge_and_phy_vol(test_data_configs):
    reg = geant4.Registry()
    gedet = make_hpge(test_data_configs + "/V99000A.json", registry=reg)

    pos = ak.Array(
        {
            "xloc": [[0, 100, 200], [100], [700, 500, 200]],
            "yloc": [[100, 0, 0], [200], [100, 300, 200]],
            "zloc": [[700, 10, 20], [100], [300, 100, 0]],
        }
    )
    tab = Table(pos)

    func_eval = {
        "mode": "function",
        "expression": "reboost.hpge.processors.distance_to_surface(hit.xloc, hit.yloc, hit.zloc, hpge, phy_vol.position.eval(), None)",
    }
    gdml_path = configs / pathlib.Path("geom.gdml")

    gdml = pyg4ometry.gdml.Reader(gdml_path).getRegistry()

    # read with the det_phy_vol_name
    phy = utils.get_phy_vol(gdml, {"phy_vol_name": "det_phy_1"}, "det001")

    assert ak.all(
        ak.num(hit.eval_expression(tab, func_eval, {}, hpge=gedet, phy_vol=phy), axis=1)
        == [3, 1, 3]
    )
