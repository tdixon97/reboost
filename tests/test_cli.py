from __future__ import annotations

from pathlib import Path

from lgdo import lh5

from reboost.cli import cli


def test_cli(tmptestdir):
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
            f"{test_file_dir}/test_files/beta_small.lh5",
        ]
    )

    glm = lh5.read("glm/det001", f"{tmptestdir}/glm.lh5").view_as("ak")
    assert glm.fields == ["evtid", "n_rows", "start_row"]

    # test cli for build_hit
    cli(
        [
            "build-hit",
            "--config",
            f"{test_file_dir}/configs/hit_config.yaml",
            "-w",
            "--glm-file",
            f"{tmptestdir}/glm.lh5",
            "--stp-file",
            f"{test_file_dir}/test_files/beta_small.lh5",
            "--hit-file",
            f"{tmptestdir}/hit.lh5",
            "--args",
            f"{test_file_dir}/configs/args.yaml",
        ]
    )

    hit1 = lh5.read("hit/det001", f"{tmptestdir}/hit.lh5").view_as("ak")
    assert set(hit1.fields) == {"evtid", "t0", "truth_energy", "active_energy", "smeared_energy"}
