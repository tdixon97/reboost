from __future__ import annotations

import copy
from pathlib import Path

import pytest
from lgdo import Array, Table, lh5

from reboost.cli import cli


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


def test_cli(tmptestdir, test_gen_lh5_flat):
    test_file_dir = Path(__file__).parent / "hit"

    # test cli for build_glm
    cli(
        [
            "build-glm",
            "--id-name",
            "evtid",
            "-w",
            "--glm-file",
            f"{tmptestdir}/glm.lh5",
            "--stp-file",
            test_gen_lh5_flat,
        ]
    )

    glm = lh5.read("glm/det1", f"{tmptestdir}/glm.lh5").view_as("ak")
    assert glm.fields == ["evtid", "n_rows", "start_row"]

    # test cli for build_hit
    cli(
        [
            "build-hit",
            "--config",
            f"{test_file_dir}/configs/reshape.yaml",
            "-w",
            "--stp-file",
            test_gen_lh5_flat,
            "--hit-file",
            f"{tmptestdir}/hit.lh5",
            "--args",
            f"{test_file_dir}/configs/args.yaml",
        ]
    )

    hit1 = lh5.read("hit/det1", f"{tmptestdir}/hit.lh5").view_as("ak")
    assert set(hit1.fields) == {"xloc", "yloc", "zloc", "dist_to_surf", "time", "edep", "evtid"}
