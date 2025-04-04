from __future__ import annotations

import json
from pathlib import Path

import awkward as ak
import legendhpges
import numpy as np
import pyg4ometry as pg4
import pygeomtools
import pytest
from dbetto import AttrsDict
from legendtestdata import LegendTestData
from lgdo import Array, Table

import reboost


@pytest.fixture(scope="session")
def test_data_configs():
    ldata = LegendTestData()
    ldata.checkout("5f9b368")
    return ldata.get_path("legend/metadata/hardware/detectors/germanium/diodes")


@pytest.fixture(scope="session")
def make_gdml(test_data_configs):
    reg = pg4.geant4.Registry()
    meta = f"{test_data_configs}/C99000A.json"
    hpge = legendhpges.make_hpge(meta, registry=reg)

    world_s = pg4.geant4.solid.Orb("World_s", 20, registry=reg, lunit="cm")
    world_l = pg4.geant4.LogicalVolume(world_s, "G4_Galactic", "World", registry=reg)
    reg.setWorld(world_l)

    # now place the two HPGe detectors in the argon
    hpge_p = pg4.geant4.PhysicalVolume(
        [0, 0, 0], [5, 0, 0, "cm"], hpge, "HPGE", world_l, registry=reg
    )
    with Path(meta).open() as file:
        metadata = json.load(file)

    hpge_p.pygeom_active_detector = pygeomtools.RemageDetectorInfo(
        "germanium",
        1,
        metadata,
    )
    pygeomtools.detectors.write_detector_auxvals(reg)

    w = pg4.gdml.Writer()
    w.addDetector(reg)
    w.write(f"{test_data_configs}/geometry.gdml")

    return f"{test_data_configs}/geometry.gdml"


def test_get_objects(test_data_configs, make_gdml):
    # check basic eval
    expression = "pyg4ometry.geant4.Registry()"
    reg = reboost.core.evaluate_object(expression, {})
    assert reg is not None
    objects = AttrsDict({"reg": reg})

    # get legend-test data
    # check if we can make a HPGe
    path = test_data_configs + r"/{DETECTOR}"
    expression = f"legendhpges.make_hpge(f'{path}.json',registry = OBJECTS.reg)"
    assert reboost.core.evaluate_object(expression, {"DETECTOR": "C99000A", "OBJECTS": objects})

    # test construction of the global objects
    # first not involving any args, second with ARGS
    args = AttrsDict(
        {
            "metadata": test_data_configs,
            "gdml": make_gdml,
            "name": "test",
        }
    )

    obj = reboost.core.get_global_objects(
        {
            "geometry": "pyg4ometry.gdml.Reader(ARGS.gdml).getRegistry()",
            "reg": "pyg4ometry.geant4.Registry()",
            "test_data": "legendtestdata.LegendTestData()",
            "meta": "dbetto.TextDB(ARGS.metadata)",
        },
        local_dict={"ARGS": args},
    )
    # check it worked
    assert sorted(obj.keys()) == ["geometry", "meta", "reg", "test_data"]

    # test getting detector objects
    expressions = {
        "meta": "pygeomtools.get_sensvol_metadata(OBJECTS.geometry, DETECTOR)",
        "hpge": "legendhpges.make_hpge(pygeomtools.get_sensvol"
        "_metadata(OBJECTS.geometry, DETECTOR),registry = OBJECTS.reg,name=ARGS.name)",
    }

    det_obj = reboost.core.get_detector_objects(["HPGE"], expressions, args, obj)

    assert next(iter(det_obj.keys())) == "HPGE"
    assert list(det_obj["HPGE"].keys()) == ["meta", "hpge"]


def test_eval_expression():
    hit_table = Table()
    hit_table.add_field("field", Array([1, 2, 3]))
    hit_table.add_field("other", Array([2, 4, 5]))
    hit_table.add_field("distances", Array([0.1, 0.6, 3]))

    # evaluate basic
    expression = "HITS.field + HITS.other"
    assert np.all(
        reboost.core.evaluate_output_column(hit_table, expression, {}, table_name="HITS").view_as(
            "np"
        )
        == [3, 6, 8]
    )

    # test also with locals
    expression = "HITS.field*args.val"
    local = {"args": AttrsDict({"val": 2})}
    assert np.all(
        reboost.core.evaluate_output_column(
            hit_table, expression, local, table_name="HITS"
        ).view_as("np")
        == [2, 4, 6]
    )

    # test with a reboost function
    expression = "reboost.math.functions.piecewise_linear_activeness(HITS.distances, fccd=args.fccd, dlf=args.dlf)"

    local = {"args": AttrsDict({"fccd": 1, "dlf": 0.5})}

    assert np.allclose(
        reboost.core.evaluate_output_column(
            hit_table, expression, local, table_name="HITS"
        ).view_as("np"),
        [0, 0.2, 1],
    )


def test_detector_mapping():
    # basic
    assert reboost.core.get_detectors_mapping("[str(i) for i in range(3)]") == {
        "0": ["0"],
        "1": ["1"],
        "2": ["2"],
    }

    assert reboost.core.get_detectors_mapping("0") == {"0": ["0"]}

    # with input name
    assert reboost.core.get_detectors_mapping(
        "[str(i) for i in range(3)]", input_detector_name="dets"
    ) == {"dets": ["0", "1", "2"]}

    # with  objects:
    objs = AttrsDict({"format": "ch"})
    assert reboost.core.get_detectors_mapping(
        "[f'{OBJECTS.format}{i}' for i in range(3)]", input_detector_name="dets", objects=objs
    ) == {"dets": ["ch0", "ch1", "ch2"]}


def test_remove_columns():
    tab = Table({"a": Array([1, 2, 3]), "b": Array([1, 2, 3])})

    tab = reboost.core.remove_columns(tab, ["a"])

    assert next(iter(tab.keys())) == "a"


def test_merge():
    tab = Table({"a": Array([2, 3])})
    output_tab = ak.Array({"a": [0, 1]})

    orig = reboost.core.merge(tab, None)
    assert np.all(orig.a == [2, 3])

    merged = reboost.core.merge(tab, output_tab)

    assert np.all(merged.a == [0, 1, 2, 3])
