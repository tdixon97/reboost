from __future__ import annotations

import pathlib

import awkward as ak
import numpy as np
import pyg4ometry
import pytest
from legendhpges.base import HPGe
from legendtestdata import LegendTestData
from lgdo import Table, lh5

from reboost.hpge import hit

configs = pathlib.Path(__file__).parent.resolve() / pathlib.Path("configs")


@pytest.fixture(scope="session")
def test_data_configs():
    ldata = LegendTestData()
    ldata.checkout("5f9b368")
    return ldata.get_path("legend/metadata/hardware/detectors/germanium/diodes")


def test_get_locals(test_data_configs):
    local_info = {"hpge": "reboost.hpge.utils.get_hpge(meta_path=meta,pars=pars,detector=detector)"}

    local_dict = hit.get_locals(
        local_info,
        pars_dict={"meta_name": "V99000A.json"},
        detector="det001",
        meta_path=test_data_configs,
    )

    assert isinstance(local_dict["hpge"], HPGe)


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
    # test get locals
    local_dict = hit.get_locals({}, pars_dict=pars)
    assert np.size(hit.eval_expression(tab, func_eval, local_dict).view_as("np")) == 5


def test_eval_with_hpge(test_data_configs):
    local_info = {"hpge": "reboost.hpge.utils.get_hpge(meta_path=meta,pars=pars,detector=detector)"}
    local_dict = hit.get_locals(
        local_info,
        pars_dict={"meta_name": "V99000A.json"},
        detector="det001",
        meta_path=test_data_configs,
    )

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

    assert ak.all(ak.num(hit.eval_expression(tab, func_eval, local_dict), axis=1) == [3, 1, 3])


def test_eval_with_hpge_and_phy_vol(test_data_configs):
    gdml_path = configs / pathlib.Path("geom.gdml")

    gdml = pyg4ometry.gdml.Reader(gdml_path).getRegistry()

    local_info = {
        "hpge": "reboost.hpge.utils.get_hpge(meta_path=meta,pars=pars,detector=detector)",
        "phy_vol": "reboost.hpge.utils.get_phy_vol(reg=reg,pars=pars,detector=detector)",
    }
    local_dict = hit.get_locals(
        local_info,
        pars_dict={"meta_name": "V99000A.json", "phy_vol_name": "det_phy_1"},
        detector="det001",
        meta_path=test_data_configs,
        reg=gdml,
    )

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

    assert ak.all(ak.num(hit.eval_expression(tab, func_eval, local_dict), axis=1) == [3, 1, 3])


# test the full processing chain
@pytest.fixture
def test_reboost_input_file(tmp_path):
    # make it large enough to be multiple groups
    rng = np.random.default_rng()
    evtid_1 = np.sort(rng.integers(int(1e5), size=(int(1e6))))
    time_1 = rng.uniform(low=0, high=1, size=(int(1e6)))
    edep_1 = rng.uniform(low=0, high=1000, size=(int(1e6)))
    pos_x_1 = rng.uniform(low=-50, high=50, size=(int(1e6)))
    pos_y_1 = rng.uniform(low=-50, high=50, size=(int(1e6)))
    pos_z_1 = rng.uniform(low=-50, high=50, size=(int(1e6)))

    # make it not divide by the buffer len
    evtid_2 = np.sort(rng.integers(int(1e5), size=(30040)))
    time_2 = rng.uniform(low=0, high=1, size=(30040))
    edep_2 = rng.uniform(low=0, high=1000, size=(30040))
    pos_x_2 = rng.uniform(low=-50, high=50, size=(30040))
    pos_y_2 = rng.uniform(low=-50, high=50, size=(30040))
    pos_z_2 = rng.uniform(low=-50, high=50, size=(30040))

    vertices_1 = ak.Array({"evtid": np.arange(int(1e5))})
    vertices_2 = ak.Array({"evtid": np.arange(int(1e5))})

    arr_1 = ak.Array(
        {
            "evtid": evtid_1,
            "time": time_1,
            "edep": edep_1,
            "xloc": pos_x_1,
            "yloc": pos_y_1,
            "zloc": pos_z_1,
        }
    )
    arr_2 = ak.Array(
        {
            "evtid": evtid_2,
            "time": time_2,
            "edep": edep_2,
            "xloc": pos_x_2,
            "yloc": pos_y_2,
            "zloc": pos_z_2,
        }
    )

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

    for output_file, input_file in zip(
        ["out.lh5", "out_rem.lh5", "out_merge.lh5"], ["file1.lh5", "file2.lh5", "file*.lh5"]
    ):
        hit.build_hit(
            str(test_reboost_input_file / output_file),
            [str(test_reboost_input_file / input_file)],
            in_field="hit",
            out_field="hit",
            proc_config=proc_config,
            pars={},
            buffer=100000,
        )

    hit.build_hit(
        str(test_reboost_input_file / "out.lh5"),
        [str(test_reboost_input_file / "file*.lh5")],
        in_field="hit",
        out_field="hit",
        proc_config=proc_config,
        pars={},
        buffer=100000,
        merge_input_files=False,
    )

    # read back in the data and check this works (no errors)

    tab = lh5.read("hit/det001", str(test_reboost_input_file / "out.lh5")).view_as("ak")
    tab_merge = lh5.read("hit/det001", str(test_reboost_input_file / "out_merge.lh5")).view_as("ak")
    tab_0 = lh5.read("hit/det001", str(test_reboost_input_file / "out_0.lh5")).view_as("ak")
    tab_1 = lh5.read("hit/det001", str(test_reboost_input_file / "out_1.lh5")).view_as("ak")

    # check size of the output
    assert len(ak.flatten(tab.evtid, axis=-1)) == int(1e6)
    assert len(ak.flatten(tab_merge.evtid, axis=-1)) == int(1e6 + 30040)
    assert len(ak.flatten(tab_0.evtid, axis=-1)) == int(1e6)
    assert len(ak.flatten(tab_1.evtid, axis=-1)) == 30040

    # check on evtid

    assert ak.all(ak.all(tab.evtid == ak.firsts(tab.evtid, axis=-1), axis=1))
    assert ak.all(ak.all(tab_merge.evtid == ak.firsts(tab_merge.evtid, axis=-1), axis=1))
    assert ak.all(ak.all(tab_0.evtid == ak.firsts(tab_0.evtid, axis=-1), axis=1))
    assert ak.all(ak.all(tab_1.evtid == ak.firsts(tab_1.evtid, axis=-1), axis=1))


def test_build_hit_some_row(test_reboost_input_file):
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

    # test asking to read too many rows
    with pytest.raises(ValueError):
        hit.build_hit(
            str(test_reboost_input_file / "out.lh5"),
            [str(test_reboost_input_file / "file1.lh5")],
            n_evtid=int(1e7),
            in_field="hit",
            out_field="hit",
            proc_config=proc_config,
            pars={},
            buffer=100000,
        )

    for n_ev, s_ev, out in zip(
        [int(1e4), int(1e5 - 1e4), int(1e5), int(1e5)],
        [0, int(1e4), 0, 1000],
        ["out_some_rows.lh5", "out_rest_rows.lh5", "out_all_file_one.lh5", "out_mix.lh5"],
    ):
        # test read only some events
        hit.build_hit(
            str(test_reboost_input_file / out),
            [
                str(test_reboost_input_file / "file1.lh5"),
                str(test_reboost_input_file / "file2.lh5"),
            ],
            n_evtid=n_ev,
            start_evtid=s_ev,
            in_field="hit",
            out_field="hit",
            proc_config=proc_config,
            pars={},
            buffer=100000,
        )

    tab_some = lh5.read("hit/det001", str(test_reboost_input_file / "out_some_rows.lh5")).view_as(
        "ak"
    )
    tab_rest = lh5.read("hit/det001", str(test_reboost_input_file / "out_rest_rows.lh5")).view_as(
        "ak"
    )

    tab_1 = lh5.read("hit/det001", str(test_reboost_input_file / "out_all_file_one.lh5")).view_as(
        "ak"
    )

    tab_merge = ak.concatenate((tab_some, tab_rest))
    assert ak.all(ak.all(tab_merge.evtid == tab_1.evtid, axis=-1))


def test_build_hit_with_locals(test_reboost_input_file, test_data_configs):
    proc_config = {
        "channels": [
            "det001",
        ],
        "outputs": ["t0", "evtid", "distance_to_surface"],
        "step_group": {
            "description": "group steps by time and evtid.",
            "expression": "reboost.hpge.processors.group_by_time(stp,window=10)",
        },
        "locals": {
            "hpge": "reboost.hpge.utils.get_hpge(meta_path=meta,pars=pars,detector=detector)",
            "phy_vol": "reboost.hpge.utils.get_phy_vol(reg=reg,pars=pars,detector=detector)",
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
            "smear_energy_sum": {
                "description": "summed energy after convolution with energy response.",
                "mode": "function",
                "expression": "reboost.hpge.processors.smear_energies(hit.truth_energy_sum,reso=pars.reso)",
            },
            "distance_to_surface": {
                "description": "distance to the nplus surface",
                "mode": "function",
                "expression": "reboost.hpge.processors.distance_to_surface(hit.xloc, hit.yloc, hit.zloc, hpge, phy_vol.position.eval(), None)",
            },
        },
    }
    gdml_path = configs / pathlib.Path("geom.gdml")
    meta_path = test_data_configs
    # complete check on the processing chain including parameters / local variables

    hit.build_hit(
        str(test_reboost_input_file / "out.lh5"),
        [str(test_reboost_input_file / "file*.lh5")],
        in_field="hit",
        out_field="hit",
        proc_config=proc_config,
        pars={"det001": {"reso": 1, "meta_name": "V99000A.json", "phy_vol_name": "det_phy_1"}},
        buffer=100000,
        merge_input_files=False,
        metadata_path=meta_path,
        gdml=gdml_path,
    )
