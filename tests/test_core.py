from __future__ import annotations

import pytest
from dbetto import AttrsDict
from legendtestdata import LegendTestData

import reboost


@pytest.fixture(scope="session")
def test_data_configs():
    ldata = LegendTestData()
    ldata.checkout("5f9b368")
    return ldata.get_path("legend/metadata/hardware/detectors/germanium/diodes")


def test_get_objects(test_data_configs):
    # check it works
    expression = "pyg4ometry.geant4.Registry()"
    reg = reboost.core.evaluate_object(expression, {})
    assert reg is not None
    objects = AttrsDict({"reg": reg})

    # get legend-test data
    # check if we can make a HPGe
    path = test_data_configs + r"/{DETECTOR}"
    expression = f"legendhpges.make_hpge(f'{path}.json',registry = OBJECTS.reg)"
    assert reboost.core.evaluate_object(expression, {"DETECTOR": "C99000A", "OBJECTS": objects})


def test_eval_expression():
    pass


def test_detector_mapping():
    pass


def test_remove_columns():
    pass
