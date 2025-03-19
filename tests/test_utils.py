from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import yaml

import reboost
from reboost.utils import get_function_string, merge_dicts


def test_search_string():
    assert reboost.utils._search_string("func.module(abc)") == ["func.module"]
    assert reboost.utils._search_string("f.c.a(abc())") == ["f.c.a", "abc"]
    assert reboost.utils._search_string("f.c.a(a(),abc)") == ["f.c.a", "a"]
    assert reboost.utils._search_string("f.c.a(a(),abc).method()") == ["f.c.a", "a", "method"]


def test_get_function_string():
    # test that the expression looks correct and the packages are properly imported

    # basic numpy
    expression = "numpy.sum(energy)"
    energy = [1, 2, 3]
    func_string, globals_dict = get_function_string(expression)
    assert func_string == "np.sum(energy)"
    assert eval(func_string, {"energy": energy}, globals_dict) == 6

    # check the import was correct
    numpy = importlib.import_module("numpy")
    assert list(globals_dict.keys()) == ["np"]
    assert globals_dict["np"] == numpy

    # now try instead awkward
    expression = "awkward.num(energy)"
    energy = [[1, 2], [1, 2, 3], [1, 2, 3, 4]]
    func_string, globals_dict = get_function_string(expression)
    assert func_string == "ak.num(energy)"
    assert list(eval(func_string, {"energy": energy}, globals_dict)) == [2, 3, 4]

    awkward = importlib.import_module("awkward")
    assert list(globals_dict.keys()) == ["ak"]
    assert globals_dict["ak"] == awkward

    # try a chain operation
    expression = "ak.sum(np.array(energy))"
    func_string, globals_dict = get_function_string(expression)
    assert func_string == expression
    assert eval(func_string, {"energy": [1, 2, 3]}, globals_dict) == 6
    assert sorted(globals_dict.keys()) == ["ak", "np"]
    assert globals_dict["ak"] == awkward
    assert globals_dict["np"] == numpy

    # try a reboost package

    expression = "reboost.math.functions.piecewise_linear_activeness(distances,fccd=1,dlf=0.5)"
    func_string, globals_dict = get_function_string(expression)

    assert func_string == expression
    assert list(globals_dict.keys()) == ["reboost"]

    distances = awkward.Array([[0.2], [2], [0.6]])

    # just check it runs
    assert eval(func_string, {"distances": distances}, globals_dict).view_as("ak")[0][0] == 0
    assert eval(func_string, {"distances": distances}, globals_dict).view_as("ak")[1][0] == 1
    assert eval(func_string, {"distances": distances}, globals_dict).view_as("ak")[2][
        0
    ] == pytest.approx(0.2)

    # try a more compliated expression
    expression = (
        "legendhpges.make_hpge(pygeomtools.get_sensvol"
        "_metadata(OBJECTS.geometry, DETECTOR),registry = OBJECT.reg,name=ARGS.name)"
    )

    func_string, globals_dict = get_function_string(expression)

    assert list(globals_dict.keys()) == ["legendhpges", "pygeomtools"]


def test_merge_dicts():
    assert merge_dicts([{"a": [1, 2, 3], "b": [2]}, {"a": [4, 5, 6], "c": [2]}]) == {
        "a": [1, 2, 3, 4, 5, 6],
        "b": [2],
        "c": [2],
    }


@pytest.fixture
def test_save_dict(tmp_path):
    data = {"a": 1, "b": {"c": 1}}

    # Dumping JSON
    with Path(tmp_path / "data.json").open("w") as json_file:
        json.dump(data, json_file, indent=2)

    # Dumping YAML
    with Path(tmp_path / "data.yaml").open("w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

    # make a csv too
    with Path(tmp_path / "data.csv").open("w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

    return tmp_path
