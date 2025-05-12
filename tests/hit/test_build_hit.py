from __future__ import annotations

from pathlib import Path

import awkward as ak
import dbetto
import pytest
from lgdo import Array, Struct, Table, lh5

import reboost
from reboost.build_glm import build_glm


@pytest.fixture(scope="module")
def test_gen_lh5(tmptestdir):
    # write a basic lh5 file

    stp_path = str(tmptestdir / "basic.lh5")
    glm_path = str(tmptestdir / "basic_glm.lh5")

    evtid = [0, 0, 1, 1, 1]
    edep = [100, 200, 10, 20, 300]  # keV
    time = [0, 1.5, 0.1, 2.1, 3.7]  # ns
    vertices = [0, 1]
    tab = Table({"evtid": Array(evtid), "edep": Array(edep), "time": Array(time)})
    lh5.write(tab, "stp/det1", stp_path, wo_mode="of")
    lh5.write(
        Table({"evtid": Array(vertices)}),
        "stp/vertices",
        stp_path,
        wo_mode="append",
    )

    build_glm(stp_path, glm_path, id_name="evtid")

    return stp_path, glm_path


def test_basic(test_gen_lh5, tmptestdir):
    stp_path, glm_path = test_gen_lh5

    reboost.build_hit.build_hit(
        f"{Path(__file__).parent}/configs/basic.yaml",
        args={},
        stp_files=stp_path,
        glm_files=glm_path,
        hit_files=f"{tmptestdir}/basic_hit.lh5",
    )

    hits = lh5.read("hit/det1", f"{tmptestdir}/basic_hit.lh5").view_as("ak")

    assert ak.all(hits.energy == [300, 330])
    assert ak.all(hits.t0 == [0, 0.1])
    assert ak.all(hits.evtid[0] == [0, 0])
    assert ak.all(hits.evtid[1] == [1, 1, 1])

    # test in memory

    hits, time_dict = reboost.build_hit.build_hit(
        f"{Path(__file__).parent}/configs/basic.yaml",
        args={},
        stp_files=stp_path,
        glm_files=glm_path,
        hit_files=None,
    )

    assert ak.all(hits["det1"].energy == [300, 330])
    assert ak.all(hits["det1"].t0 == [0, 0.1])
    assert ak.all(hits["det1"].evtid[0] == [0, 0])
    assert ak.all(hits["det1"].evtid[1] == [1, 1, 1])

    assert list(time_dict.keys()) == ["global_objects", "geds"]
    assert list(time_dict["geds"].keys()) == [
        "detector_objects",
        "read",
        "hit_layout",
        "expressions",
    ]
    assert list(time_dict["geds"]["read"].keys()) == ["glm", "stp"]
    assert list(time_dict["geds"]["expressions"].keys()) == ["t0", "first_evtid", "energy"]


def test_full_chain(tmptestdir):
    build_glm(
        f"{Path(__file__).parent}/test_files/beta_small.lh5",
        str(tmptestdir / "beta_small_glm.lh5"),
        id_name="evtid",
    )

    args = dbetto.AttrsDict(
        {
            "gdml": f"{Path(__file__).parent}/configs/geom.gdml",
            "pars": f"{Path(__file__).parent}/configs/pars.yaml",
        }
    )

    _, time_dict = reboost.build_hit.build_hit(
        f"{Path(__file__).parent}/configs/hit_config.yaml",
        args=args,
        stp_files=f"{Path(__file__).parent}/test_files/beta_small.lh5",
        glm_files=str(tmptestdir / "beta_small_glm.lh5"),
        hit_files=f"{Path(__file__).parent}/test_files/beta_small_hit.lh5",
        overwrite=True,
    )
    hits = lh5.read("hit", f"{Path(__file__).parent}/test_files/beta_small_hit.lh5")

    assert isinstance(hits, Struct)

    assert set(hits["det001"].view_as("ak").fields) == {
        "evtid",
        "t0",
        "truth_energy",
        "active_energy",
        "smeared_energy",
    }
    assert set(hits["det002"].view_as("ak").fields) == {
        "evtid",
        "t0",
        "truth_energy",
        "active_energy",
        "smeared_energy",
    }
