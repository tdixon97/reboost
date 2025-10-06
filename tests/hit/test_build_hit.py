from __future__ import annotations

import copy
from pathlib import Path

import awkward as ak
import dbetto
import h5py
import pytest
from lgdo import Array, Struct, Table, VectorOfVectors, lh5

import reboost


@pytest.fixture(scope="module")
def test_gen_lh5(tmptestdir):
    # write a basic lh5 file

    stp_path = str(tmptestdir / "basic.lh5")

    data = {}
    data["evtid"] = Array([0, 1])
    data["edep"] = VectorOfVectors([[100, 200], [10, 20, 300]])  # keV
    data["time"] = VectorOfVectors([[0, 1.5], [0.1, 2.1, 3.7]])  # ns

    data["xloc"] = VectorOfVectors([[0.01, 0.02], [0.001, 0.003, 0.005]])  # m
    data["yloc"] = VectorOfVectors([[0.01, 0.02], [0.001, 0.003, 0.005]])  # m
    data["zloc"] = VectorOfVectors([[0.04, 0.02], [0.001, 0.023, 0.005]])  # m
    data["dist_to_surf"] = VectorOfVectors([[0.04, 0.02], [0.011, 0.003, 0.051]])  # m

    vertices = [0, 1]
    tab = Table(data)
    tab2 = copy.deepcopy(tab)

    lh5.write(tab, "stp/det1", stp_path, wo_mode="of")
    lh5.write(tab2, "stp/det2", stp_path, wo_mode="append")
    lh5.write(
        Table({"evtid": Array(vertices)}),
        "vtx",
        stp_path,
        wo_mode="append",
    )

    return stp_path


@pytest.fixture(scope="module")
def test_gen_lh5_flat(tmptestdir):
    # write a basic lh5 file

    stp_path = str(tmptestdir / "basic_flat.lh5")

    data = {}
    data["evtid"] = Array([0, 0, 1, 1, 1])
    data["edep"] = Array([100, 200, 10, 20, 300])  # keV
    data["time"] = Array([0, 1.5, 0.1, 2.1, 3.7])  # ns

    data["xloc"] = Array([0.01, 0.02, 0.001, 0.003, 0.005])  # m
    data["yloc"] = Array([0.01, 0.02, 0.001, 0.003, 0.005])  # m
    data["zloc"] = Array([0.04, 0.02, 0.001, 0.023, 0.005])  # m
    data["dist_to_surf"] = Array([0.04, 0.02, 0.011, 0.003, 0.051])  # m

    vertices = [0, 1]
    tab = Table(data)
    tab2 = copy.deepcopy(tab)

    lh5.write(tab, "stp/det1", stp_path, wo_mode="of")
    lh5.write(tab2, "stp/det2", stp_path, wo_mode="append")
    lh5.write(
        Table({"evtid": Array(vertices)}),
        "vtx",
        stp_path,
        wo_mode="append",
    )

    return stp_path


@pytest.fixture(scope="module")
def test_gen_lh5_scint(tmptestdir):
    # write a basic lh5 file with scintillator

    stp_path = str(tmptestdir / "basic_scint.lh5")

    data = {}
    data["evtid"] = Array([0, 1])
    data["edep"] = VectorOfVectors([[100, 200], [10, 20, 300]])  # keV
    data["time"] = VectorOfVectors([[0, 1.5], [0.1, 2.1, 3.7]])  # ns

    data["xloc"] = VectorOfVectors([[0.01, 0.02], [0.001, 0.003, 0.005]])  # m
    data["yloc"] = VectorOfVectors([[0.01, 0.02], [0.001, 0.003, 0.005]])  # m
    data["zloc"] = VectorOfVectors([[0.04, 0.02], [0.001, 0.023, 0.005]])  # m

    data["particle"] = VectorOfVectors([[11, 11], [11, 11, 11]])

    vertices = [0, 1]
    tab = Table(data)

    lh5.write(tab, "stp/det001", stp_path, wo_mode="of")
    lh5.write(
        Table({"evtid": Array(vertices)}),
        "vtx",
        stp_path,
        wo_mode="append",
    )

    return stp_path


def test_reshape(test_gen_lh5_flat, tmptestdir):
    outfile = f"{tmptestdir}/basic_hit_reshaped.lh5"

    reboost.build_hit(
        f"{Path(__file__).parent}/configs/reshape.yaml",
        args={},
        stp_files=test_gen_lh5_flat,
        glm_files=None,
        hit_files=outfile,
        out_field="stp",
        overwrite=True,
    )

    data = {}
    data["evtid"] = VectorOfVectors([[0, 0], [1, 1, 1]])
    data["edep"] = VectorOfVectors([[100, 200], [10, 20, 300]])  # keV
    data["time"] = VectorOfVectors([[0, 1.5], [0.1, 2.1, 3.7]])  # ns

    data["xloc"] = VectorOfVectors([[0.01, 0.02], [0.001, 0.003, 0.005]])  # m
    data["yloc"] = VectorOfVectors([[0.01, 0.02], [0.001, 0.003, 0.005]])  # m
    data["zloc"] = VectorOfVectors([[0.04, 0.02], [0.001, 0.023, 0.005]])  # m
    data["dist_to_surf"] = VectorOfVectors([[0.04, 0.02], [0.011, 0.003, 0.051]])  # m
    tab = Table(data)

    output = lh5.read("stp/det1", outfile)

    # check the outputs
    assert output == tab
    assert lh5.read("vtx", outfile) == Table({"evtid": Array([0, 1])})


def test_only_forward(test_gen_lh5_flat, tmptestdir):
    outfile = f"{tmptestdir}/basic_hit_foward_only.lh5"

    for _ in range(2):  # test that overwriting the output file works.
        reboost.build_hit(
            f"{Path(__file__).parent}/configs/foward_only.yaml",
            args={},
            stp_files=test_gen_lh5_flat,
            glm_files=None,
            hit_files=outfile,
            out_field="stp",
            overwrite=True,
        )

    # check the outputs
    assert lh5.read("vtx", outfile) == Table({"evtid": Array([0, 1])})


def test_basic(test_gen_lh5, tmptestdir):
    outfile = f"{tmptestdir}/basic_hit.lh5"

    reboost.build_hit(
        f"{Path(__file__).parent}/configs/basic.yaml",
        args={},
        stp_files=test_gen_lh5,
        glm_files=None,
        hit_files=outfile,
        overwrite=True,
    )

    assert lh5.ls(outfile) == ["hit", "vtx"]

    with h5py.File(outfile) as h5f:
        assert (
            h5f["/hit/det1/energy"].id.get_create_plist().get_filter(0)[3]
            == b"Zstandard compression: http://www.zstd.net"
        )

    hits = lh5.read("hit/det1", outfile).view_as("ak")

    assert ak.all(hits.energy == [300, 330])
    assert ak.all(hits.t0 == [0, 0.1])
    assert hits.evtid[0] == 0
    assert hits.evtid[1] == 1

    assert len(hits) == 2

    # test in memory

    hits, time_dict = reboost.build_hit(
        f"{Path(__file__).parent}/configs/basic.yaml",
        args={},
        stp_files=test_gen_lh5,
        glm_files=None,
        hit_files=None,
    )

    assert ak.all(hits["det1"].energy == [300, 330])
    assert ak.all(hits["det1"].t0 == [0, 0.1])
    assert ak.all(hits["det1"].evtid[0] == [0, 0])
    assert ak.all(hits["det1"].evtid[1] == [1, 1, 1])

    assert set(time_dict.keys()) == {"global_objects", "geds"}
    assert set(time_dict["geds"].keys()) == {
        "detector_objects",
        "read",
        "conv",
        "expressions",
    }
    assert set(time_dict["geds"]["read"].keys()) == {"stp"}
    assert set(time_dict["geds"]["expressions"].keys()) == {"t0", "energy"}


def test_file_merging(test_gen_lh5, tmptestdir):
    outfile = f"{tmptestdir}/basic_hit_merged.lh5"

    reboost.build_hit(
        f"{Path(__file__).parent}/configs/basic.yaml",
        args={},
        stp_files=[test_gen_lh5, test_gen_lh5],
        glm_files=None,
        hit_files=outfile,
        overwrite=True,
    )

    assert lh5.ls(outfile) == ["hit", "vtx"]

    hits = lh5.read("hit/det1", outfile).view_as("ak")

    assert len(hits) == 4


def test_multi_file(test_gen_lh5, tmptestdir):
    outfile = [f"{tmptestdir}/basic_hit_t0.lh5", f"{tmptestdir}/basic_hit_t1.lh5"]

    reboost.build_hit(
        f"{Path(__file__).parent}/configs/basic.yaml",
        args={},
        stp_files=[test_gen_lh5, test_gen_lh5],
        glm_files=None,
        hit_files=outfile,
        overwrite=True,
    )

    for file in outfile:
        assert lh5.ls(file) == ["hit", "vtx"]

        hits = lh5.read("hit/det1", file).view_as("ak")

        assert len(hits) == 2


def test_overwrite(test_gen_lh5, tmptestdir):
    # test with two output files
    outfile = [f"{tmptestdir}/basic_hit_t0.lh5", f"{tmptestdir}/basic_hit_t1.lh5"]

    reboost.build_hit(
        f"{Path(__file__).parent}/configs/basic.yaml",
        args={},
        stp_files=[test_gen_lh5, test_gen_lh5],
        glm_files=None,
        hit_files=outfile,
        overwrite=True,
    )
    for file in outfile:
        assert lh5.ls(file) == ["hit", "vtx"]

        hits = lh5.read("hit/det1", file).view_as("ak")

        assert len(hits) == 2

    outfile = f"{tmptestdir}/basic_hit_merged.lh5"

    reboost.build_hit(
        f"{Path(__file__).parent}/configs/basic.yaml",
        args={},
        stp_files=[test_gen_lh5, test_gen_lh5],
        glm_files=None,
        hit_files=outfile,
        overwrite=True,
    )
    assert lh5.ls(outfile) == ["hit", "vtx"]
    hits = lh5.read("hit/det1", outfile).view_as("ak")
    assert len(hits) == 4


def test_full_chain(test_gen_lh5, tmptestdir):
    args = dbetto.AttrsDict(
        {
            "gdml": f"{Path(__file__).parent}/configs/geom.gdml",
            "pars": f"{Path(__file__).parent}/configs/pars.yaml",
        }
    )

    reboost.build_hit(
        f"{Path(__file__).parent}/configs/hit_config.yaml",
        args=args,
        stp_files=test_gen_lh5,
        glm_files=None,
        hit_files=str(tmptestdir / "beta_small_hit.lh5"),
        overwrite=True,
    )
    hits = lh5.read("hit", str(tmptestdir / "beta_small_hit.lh5"))

    assert isinstance(hits, Struct)

    assert set(hits["det1"].view_as("ak").fields) == {
        "evtid",
        "t0",
        "truth_energy",
        "active_energy",
        "smeared_energy",
    }
    assert set(hits["det2"].view_as("ak").fields) == {
        "evtid",
        "t0",
        "truth_energy",
        "active_energy",
        "smeared_energy",
    }
    # also check the processing of the vtx table
    assert hits["vtx"] == Table({"evtid": Array([0, 1])})


def test_spms(test_gen_lh5_scint, tmptestdir):
    from reboost.optmap import OpticalMap

    # create a simple test map
    map_file = f"{tmptestdir}/spms_hit_map.lh5"
    m = OpticalMap.create_empty(
        "_11", {"range_in_m": [[-1, 1], [-1, 1], [-1, 1]], "bins": [1, 1, 1]}
    )
    m._nda(m.h_vertex)[0, 0, 0] = 100
    m._nda(m.h_hits)[0, 0, 0] = 10
    m.create_probability()
    m.write_lh5(map_file, "_11", "overwrite_file")
    m.write_lh5(map_file, "_12", "write_safe")

    outfile = f"{tmptestdir}/spms_hit.lh5"
    reboost.build_hit(
        f"{Path(__file__).parent}/configs/spms.yaml",
        args={"optmap_path": map_file},
        stp_files=test_gen_lh5_scint,
        glm_files=None,
        hit_files=outfile,
        overwrite=True,
    )
    hits = lh5.read("hit", outfile)

    assert "S001" in hits
    assert "S002" in hits

    assert isinstance(hits["S001"]["pe_times_lar"], VectorOfVectors)
