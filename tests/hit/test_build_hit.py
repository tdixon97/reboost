from __future__ import annotations

from pathlib import Path

import awkward as ak
import pytest
from lgdo import Array, Table, lh5

import reboost
from reboost.build_glm import build_glm


@pytest.fixture
def test_gen_lh5(tmp_path):
    # write a basic lh5 file

    evtid = [0, 0, 1, 1, 1]
    edep = [100, 200, 10, 20, 300]  # keV
    time = [0, 1.5, 0.1, 2.1, 3.7]  # ns
    vertices = [0, 1]
    tab = Table({"evtid": Array(evtid), "edep": Array(edep), "time": Array(time)})
    lh5.write(tab, "stp/det1", str(tmp_path / "basic.lh5"), wo_mode="of")
    lh5.write(
        Table({"evtid": Array(vertices)}),
        "stp/vertices",
        str(tmp_path / "basic.lh5"),
        wo_mode="append",
    )

    build_glm(str(tmp_path / "basic.lh5"), str(tmp_path / "basic_glm.lh5"), id_name="evtid")

    return str(tmp_path)


def test_basic(test_gen_lh5):
    reboost.build_hit.build_hit(
        f"{Path(__file__).parent}/configs/basic.yaml",
        args={},
        stp_files=f"{test_gen_lh5}/basic.lh5",
        glm_files=f"{test_gen_lh5}/basic_glm.lh5",
        hit_files=f"{test_gen_lh5}/basic_hit.lh5",
    )

    hits = lh5.read("det1/hit", f"{test_gen_lh5}/basic_hit.lh5").view_as("ak")

    assert ak.all(hits.energy == [300, 330])
    assert ak.all(hits.t0 == [0, 0.1])
    assert ak.all(hits.evtid[0] == [0, 0])
    assert ak.all(hits.evtid[1] == [1, 1, 1])
