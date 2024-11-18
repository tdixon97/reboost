from __future__ import annotations

import json
import pathlib

import awkward as ak
import numpy as np
import pyg4ometry
import pytest
import yaml
from legendhpges.base import HPGe
from legendtestdata import LegendTestData
from lgdo import Array, Table, lh5

from reboost.hpge.utils import (
    _merge_arrays,
    dict2tuple,
    get_file_list,
    get_files_to_read,
    get_global_evtid,
    get_global_evtid_range,
    get_hpge,
    get_include_chunk,
    get_num_simulated,
    get_phy_vol,
    load_dict,
)

configs = pathlib.Path(__file__).parent.resolve() / pathlib.Path("configs")


def test_merge():
    ak_obj = ak.Array({"evtid": [1, 1, 1, 1, 2, 2, 3], "edep": [100, 50, 1000, 20, 100, 200, 10]})
    bufer_rows = ak.Array({"evtid": [1, 1], "edep": [60, 50]})

    # should only remove the last element
    merged_idx_0, buffer_0, mode = _merge_arrays(ak_obj, None, 0, 100, True)

    assert ak.all(merged_idx_0.evtid == [1, 1, 1, 1, 2, 2])
    assert ak.all(merged_idx_0.edep == [100, 50, 1000, 20, 100, 200])

    assert ak.all(buffer_0.evtid == [3])
    assert ak.all(buffer_0.edep == [10])

    # delete input file
    assert mode == "of"

    # if delete input is false it should be appended
    _, _, mode = _merge_arrays(ak_obj, None, 0, 100, False)
    assert mode == "append"

    # now if idx isn't 0 or the max_idx should add the buffer and remove the end

    merged_idx, buffer, mode = _merge_arrays(ak_obj, bufer_rows, 2, 100, True)

    assert ak.all(merged_idx.evtid == [1, 1, 1, 1, 1, 1, 2, 2])
    assert ak.all(merged_idx.edep == [60, 50, 100, 50, 1000, 20, 100, 200])

    assert ak.all(buffer.evtid == [3])
    assert ak.all(buffer.edep == [10])

    assert mode == "append"

    # now for the final index just adds the buffer

    merged_idx_end, buffer_end, mode = _merge_arrays(ak_obj, bufer_rows, 100, 100, True)

    assert ak.all(merged_idx_end.evtid == [1, 1, 1, 1, 1, 1, 2, 2, 3])
    assert ak.all(merged_idx_end.edep == [60, 50, 100, 50, 1000, 20, 100, 200, 10])

    assert buffer_end is None


@pytest.fixture
def file_fixture(tmp_path):
    # Create a simple YAML file
    data = {"det": 1}
    yaml_file = tmp_path / "data.yaml"
    with pathlib.Path.open(yaml_file, "w") as yf:
        yaml.dump(data, yf)

    json_file = tmp_path / "data.json"
    with pathlib.Path.open(json_file, "w") as jf:
        json.dump(data, jf)

    # Create a simple TXT file
    txt_file = tmp_path / "data.txt"
    with pathlib.Path.open(txt_file, "w") as tf:
        tf.write("Some text.\n")

    # Return paths for the test functions
    return {"yaml_file": yaml_file, "json_file": json_file, "txt_file": txt_file}


def test_read(file_fixture):
    json_dict = load_dict(file_fixture["json_file"], None)
    assert json_dict["det"] == 1

    yaml_dict = load_dict(file_fixture["yaml_file"], None)
    assert yaml_dict["det"] == 1

    with pytest.raises(NotImplementedError):
        load_dict(file_fixture["txt_file"], None)


@pytest.fixture
def file_list(tmp_path):
    # make a list of files
    for i in range(5):
        data = {"det": i}

        # make a json file
        json_file = tmp_path / f"data_{i}.json"
        with pathlib.Path.open(json_file, "w") as jf:
            json.dump(data, jf)

        # and a text file
        txt_file = tmp_path / f"data_{i}.txt"
        with pathlib.Path.open(txt_file, "w") as tf:
            tf.write("Some text.\n")
    return pathlib.Path(tmp_path)


def test_get_file_list(file_list):
    first_file_list = get_file_list(str(pathlib.Path(file_list) / "data_0.json"))
    assert len(first_file_list) == 1

    json_file_list = get_file_list(str(pathlib.Path(file_list) / "data*.json"))
    assert len(json_file_list) == 5
    json_file_list_repeat = get_file_list(
        [str(pathlib.Path(file_list) / "data*.json"), str(pathlib.Path(file_list) / "data*.json")]
    )
    assert len(json_file_list_repeat) == 5
    all_file_list = get_file_list([str(pathlib.Path(file_list) / "data*")])
    assert len(all_file_list) == 10


@pytest.fixture(scope="session")
def test_data_configs():
    ldata = LegendTestData()
    ldata.checkout("5f9b368")
    return ldata.get_path("legend/metadata/hardware/detectors/germanium/diodes")


def test_get_hpge(test_data_configs):
    # specify name in pars
    hpge = get_hpge(str(test_data_configs), dict2tuple({"meta_name": "C99000A.json"}), "det001")
    assert isinstance(hpge, HPGe)

    # now read without metaname
    hpge_ic = get_hpge(str(test_data_configs), dict2tuple({}), "V99000A")
    assert isinstance(hpge_ic, HPGe)


def test_get_phy_vol():
    gdml_path = configs / pathlib.Path("geom.gdml")

    gdml = pyg4ometry.gdml.Reader(gdml_path).getRegistry()

    # read with the det_phy_vol_name
    phy = get_phy_vol(gdml, dict2tuple({"phy_vol_name": "det_phy_1"}), "det001")

    assert isinstance(phy, pyg4ometry.geant4.PhysicalVolume)

    # read without
    phy = get_phy_vol(gdml, dict2tuple({}), "det_phy_0")
    assert isinstance(phy, pyg4ometry.geant4.PhysicalVolume)


@pytest.fixture
def test_lh5_files(tmp_path):
    n1 = 14002
    tab1 = Table(size=n1)
    tab1.add_field("a", Array(np.ones(n1)))
    lh5.write(tab1, "hit/vertices", tmp_path / "file1.lh5", wo_mode="of")

    n2 = 25156
    tab2 = Table(size=n2)
    tab2.add_field("a", Array(np.ones(n2)))

    lh5.write(tab2, "hit/vertices", tmp_path / "file2.lh5", wo_mode="of")

    n3 = int(1e7)
    tab3 = Table(size=n3)
    tab3.add_field("a", Array(np.ones(n3)))

    lh5.write(tab3, "hit/vertices", tmp_path / "file3.lh5", wo_mode="of")

    return tmp_path


def test_get_n_sim(test_lh5_files):
    # single file
    n1 = get_num_simulated([str(test_lh5_files / "file1.lh5")])
    assert n1 == [14002]

    # two files
    n12 = get_num_simulated([str(test_lh5_files / "file1.lh5"), str(test_lh5_files / "file2.lh5")])
    assert n12 == [14002, 25156]

    # length > buffer
    n123 = get_num_simulated(
        [
            str(test_lh5_files / "file1.lh5"),
            str(test_lh5_files / "file2.lh5"),
            str(test_lh5_files / "file3.lh5"),
        ]
    )
    assert n123 == [14002, 25156, int(1e7)]


def test_global_evtid_range():
    # raise exception if n_evtid is too large
    with pytest.raises(ValueError):
        get_global_evtid_range(100, 1000, 200)

    # test that we get the right ranges
    assert get_global_evtid_range(200, 5, 2000) == (200, 204)
    assert get_global_evtid_range(200, None, 2000) == (200, 1999)


def test_get_global_evtid():
    # single file
    first_evtid = 0
    vertices = [0, 1, 2, 3, 4, 5]
    input_evtid = [2, 3, 4, 5, 5, 5]
    obj = ak.Array({"evtid": input_evtid})
    assert np.all(get_global_evtid(first_evtid, obj, vertices).global_evtid == input_evtid)

    # now if we only have some vertices
    vertices = [0, 2, 4, 6, 8, 10]
    input_evtid = [4, 6, 8, 10, 10, 10]
    obj = ak.Array({"evtid": input_evtid})
    assert np.all(
        get_global_evtid(first_evtid, obj, vertices).global_evtid == np.array(input_evtid) / 2.0
    )


def test_get_files_to_read():
    n_sim = [1000, 1200, 200, 5000]
    n_sim = np.concatenate([[0], np.cumsum(n_sim)])

    # read all evtid and thus all files
    assert np.all(get_files_to_read(n_sim, start_glob_evtid=0, end_glob_evtid=7399) == [0, 1, 2, 3])

    # all of file 0
    assert np.all(get_files_to_read(n_sim, start_glob_evtid=0, end_glob_evtid=999) == [0])

    # and some of file 1
    assert np.all(get_files_to_read(n_sim, start_glob_evtid=0, end_glob_evtid=1000) == [0, 1])

    # some of file 0, 1 and 2
    assert np.all(get_files_to_read(n_sim, start_glob_evtid=0, end_glob_evtid=2200) == [0, 1, 2])

    # only file 1 and 2
    assert np.all(get_files_to_read(n_sim, start_glob_evtid=1000, end_glob_evtid=2300) == [1, 2])

    # only file 3
    assert np.all(get_files_to_read(n_sim, start_glob_evtid=2400, end_glob_evtid=5000) == [3])


def test_skip_chunk():
    evtid = ak.Array([4, 5, 10, 30])

    assert get_include_chunk(evtid, start_glob_evtid=0, end_glob_evtid=35)
    assert get_include_chunk(evtid, start_glob_evtid=0, end_glob_evtid=4)
    assert get_include_chunk(evtid, start_glob_evtid=30, end_glob_evtid=33)
    assert not get_include_chunk(evtid, start_glob_evtid=0, end_glob_evtid=3)
    assert not get_include_chunk(evtid, start_glob_evtid=31, end_glob_evtid=100)
