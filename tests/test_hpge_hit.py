from __future__ import annotations

import pathlib

import awkward as ak
import numpy as np
import pyg4ometry
import pytest
from legendhpges import make_hpge
from legendtestdata import LegendTestData
from lgdo import Table, lh5
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


# test the full processing chain
@pytest.fixture
def test_reboost_input_file(tmp_path):
    # make it large enough to be multiple groups
    rng = np.random.default_rng()
    evtid_1 = np.sort(rng.integers(int(1e5), size=(int(1e6))))
    time_1 = rng.uniform(low=0, high=1, size=(int(1e6)))
    edep_1 = rng.uniform(low=0, high=1000, size=(int(1e6)))

    # make it not divide by the buffer len
    evtid_2 = np.sort(rng.integers(int(1e5), size=(30040)))
    time_2 = rng.uniform(low=0, high=1, size=(30040))
    edep_2 = rng.uniform(low=0, high=1000, size=(30040))

    vertices_1 = ak.Array({"evtid": np.arange(int(1e5))})
    vertices_2 = ak.Array({"evtid": np.arange(30040)})

    arr_1 = ak.Array({"evtid": evtid_1, "time": time_1, "edep": edep_1})
    arr_2 = ak.Array({"evtid": evtid_2, "time": time_2, "edep": edep_2})

    lh5.write(Table(vertices_1), "hit/vertices", tmp_path / "file1.lh5", wo_mode="of")
    lh5.write(Table(vertices_2), "hit/vertices", tmp_path / "file2.lh5", wo_mode="of")

    lh5.write(Table(arr_1), "hit/det001", tmp_path / "file1.lh5", wo_mode="append")
    lh5.write(Table(arr_2), "hit/det001", tmp_path / "file2.lh5", wo_mode="append")

    return tmp_path


def test_build_hit(test_reboost_input_file):
    # first just one file no pars

    proc_config = {
        "channels": [
            "det001",
        ],
        "outputs": ["t0", "evtid"],
        "step_group": {
            "description": "group steps by time and evtid.",
            "expression": "reboost.hpge.processors.group_by_time(stp,window=10)",
        },
        "operations": {
            "t0": {
                "description": "first time in the hit.",
                "mode": "eval",
                "expression": "ak.fill_none(ak.firsts(hit.time,axis=-1),np.nan)",
            },
            "truth_energy_sum": {
                "description": "truth summed energy in the hit.",
                "mode": "eval",
                "expression": "ak.sum(hit.edep,axis=-1)",
            },
        },
    }

    hit.build_hit(
        str(test_reboost_input_file / "out.lh5"),
        [str(test_reboost_input_file / "file1.lh5")],
        "hit",
        "hit",
        proc_config,
        {},
        buffer=100000,
    )
    hit.build_hit(
        str(test_reboost_input_file / "out_rem.lh5"),
        [str(test_reboost_input_file / "file2.lh5")],
        "hit",
        "hit",
        proc_config,
        {},
        buffer=10000,
    )

    # now with wildcard
    hit.build_hit(
        str(test_reboost_input_file / "out_merge.lh5"),
        [str(test_reboost_input_file / "file*.lh5")],
        "hit",
        "hit",
        proc_config,
        {},
        buffer=100000,
        merge_input_files=True,
    )
    hit.build_hit(
        str(test_reboost_input_file / "out.lh5"),
        [str(test_reboost_input_file / "file*.lh5")],
        "hit",
        "hit",
        proc_config,
        {},
        buffer=100000,
        merge_input_files=False,
    )

    # read back in the data and check this works (no errors)

    tab = lh5.read("hit/det001",str(test_reboost_input_file / "out.lh5")).view_as("ak")
    tab_merge = lh5.read("hit/det001",str(test_reboost_input_file / "out_merge.lh5")).view_as("ak")
    tab_0  = lh5.read("hit/det001",str(test_reboost_input_file / "out_0.lh5")).view_as("ak")
    tab_1  = lh5.read("hit/det001",str(test_reboost_input_file / "out_1.lh5")).view_as("ak")

    # check size of the output
    assert len(ak.flatten(tab.evtid,axis=-1))==int(1e6)
    assert len(ak.flatten(tab_merge.evtid,axis=-1))==int(1e6+30040)
    assert len(ak.flatten(tab_0.evtid,axis=-1))==int(1e6)
    assert len(ak.flatten(tab_1.evtid,axis=-1))==int(30040)

    # check on evtid

    assert ak.all(ak.all(tab.evtid == ak.firsts(tab.evtid,axis=-1), axis=1))
    assert ak.all(ak.all(tab_merge.evtid == ak.firsts(tab_merge.evtid,axis=-1), axis=1))
    assert ak.all(ak.all(tab_0.evtid == ak.firsts(tab_0.evtid,axis=-1), axis=1))
    assert ak.all(ak.all(tab_1.evtid == ak.firsts(tab_1.evtid,axis=-1), axis=1))