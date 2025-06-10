from __future__ import annotations

import awkward as ak
import numpy as np
import pytest
from lgdo import Table, lh5

from reboost.build_glm import build_glm, get_glm_rows, get_stp_evtids
from reboost.build_hit import GLMIterator


# test the basic (awkward operations) to get the glm rows
def test_get_glm_rows():
    # some basic cases
    stp_evtids = [0, 0, 0, 1, 1, 1, 6, 6, 7]
    vert = [0, 1, 6, 7]
    glm = get_glm_rows(stp_evtids, vert, start_row=0)

    assert ak.all(glm.evtid == [0, 1, 6, 7])
    assert ak.all(glm.n_rows == [3, 3, 2, 1])
    assert ak.all(glm.start_row == [0, 3, 6, 8])

    # test with a different start row

    glm = get_glm_rows(stp_evtids, vert, start_row=999)
    assert ak.all(glm.evtid == [0, 1, 6, 7])
    assert ak.all(glm.n_rows == [3, 3, 2, 1])
    assert ak.all(glm.start_row == [999, 1002, 1005, 1007])

    # test discard steps

    vert = [1, 6]
    glm = get_glm_rows(stp_evtids, vert, start_row=0)
    assert ak.all(glm.evtid == [1, 6])
    assert ak.all(glm.n_rows == [3, 2])
    assert ak.all(glm.start_row == [3, 6])

    # test gracefully fails

    # steps not in the vert table will cause it to fail
    with pytest.raises(ValueError):
        get_glm_rows(stp_evtids=[1, 3], vert=[1, 2, 4])

    # steps must be sorted
    with pytest.raises(ValueError):
        get_glm_rows(stp_evtids=[1, 3, 2], vert=[1, 2])

    # vertex evtids must be sorted
    with pytest.raises(ValueError):
        get_glm_rows(stp_evtids=[1, 3, 2], vert=[1, 2, 0])


# create some example inputs
def test_data_files(tmptestdir):
    rng = np.random.default_rng()

    # simple every evtid in vertices
    vertex_evtid = ak.Array({"evtid": np.arange(10000)})
    lh5.write(Table(vertex_evtid), "vtx", tmptestdir / "simple_test.lh5", wo_mode="of")

    # make some simple stp file
    steps_1 = ak.Array({"evtid": np.sort(rng.integers(0, 10000, size=21082))})
    steps_2 = ak.Array({"evtid": np.sort(rng.integers(0, 1000, size=1069))})

    lh5.write(Table(steps_1), "stp/det1", tmptestdir / "simple_test.lh5", wo_mode="append")
    lh5.write(Table(steps_2), "stp/det2", tmptestdir / "simple_test.lh5", wo_mode="append")

    # file with some gaps (multithreaded mode)

    vertex_evtid = ak.Array({"evtid": np.sort(np.unique(rng.integers(0, 200000, size=10000)))})
    lh5.write(Table(vertex_evtid), "vtx", tmptestdir / "gaps_test.lh5", wo_mode="of")

    # make some simple stp file
    steps_1 = ak.Array({"evtid": np.sort(rng.choice(vertex_evtid.evtid, size=21082))})
    steps_2 = ak.Array({"evtid": np.sort(rng.choice(vertex_evtid.evtid, size=1069))})

    lh5.write(Table(steps_1), "stp/det1", tmptestdir / "gaps_test.lh5", wo_mode="append")
    lh5.write(Table(steps_2), "stp/det2", tmptestdir / "gaps_test.lh5", wo_mode="append")


def test_read_stp_rows(tmptestdir):
    # check reading from the start everything
    start_row, chunk_start, evtids = get_stp_evtids(
        "stp/det1",
        str(tmptestdir / "simple_test.lh5"),
        "evtid",
        start_row=0,
        last_vertex_evtid=10000,
        stp_buffer=1000,
    )
    # read the evtid directly to compare
    evtids_read = lh5.read_as("stp/det1/evtid", str(tmptestdir / "simple_test.lh5"), "np")
    assert chunk_start == 0
    assert np.all(evtids == evtids_read)

    # check the breaking
    # the number of evtids less than 1200 is index
    index = sum(evtids_read < 1200)
    start_row, chunk_start, evtids = get_stp_evtids(
        "stp/det1",
        str(tmptestdir / "simple_test.lh5"),
        "evtid",
        start_row=0,
        last_vertex_evtid=1200,
        stp_buffer=1000,
    )
    # check we read far enough
    assert len(evtids) > index
    assert chunk_start == 0
    assert start_row == np.floor(index / 1000) * 1000

    # check updated start row

    start_row, chunk_start, evtids = get_stp_evtids(
        "stp/det1",
        str(tmptestdir / "simple_test.lh5"),
        "evtid",
        start_row=300,
        last_vertex_evtid=10000,
        stp_buffer=1000,
    )
    # first chunk should be 300
    assert chunk_start == 300
    assert len(evtids) == len(evtids_read) - 300

    # check reading of last chunk

    start_row, chunk_start, evtids = get_stp_evtids(
        "stp/det1",
        str(tmptestdir / "simple_test.lh5"),
        "evtid",
        start_row=21050,
        last_vertex_evtid=10000,
        stp_buffer=1000,
    )
    assert len(evtids) == len(evtids_read) - 21050
    assert start_row == 21050


def test_build_glm(tmptestdir):
    # produce directly glm without iteration
    # try with different buffers

    for buffer in [71, 100, 1000, 2000, 40000]:
        # two files (no gaps and gaps)
        for test in ["simple", "gaps"]:
            evtids = lh5.read_as("vtx/evtid", str(tmptestdir / f"{test}_test.lh5"), "np")

            evtids1_read = lh5.read_as("stp/det1/evtid", str(tmptestdir / f"{test}_test.lh5"), "np")
            evtids2_read = lh5.read_as("stp/det2/evtid", str(tmptestdir / f"{test}_test.lh5"), "np")
            # check both returning and saving
            for glm_file in [str(tmptestdir / f"{test}_glm.lh5"), None]:
                glm = build_glm(
                    str(tmptestdir / f"{test}_test.lh5"),
                    glm_file,
                    id_name="evtid",
                    evtid_buffer=1000,
                    stp_buffer=buffer,
                )

                if glm_file is not None:
                    glm1 = lh5.read("glm/det1", glm_file).view_as("ak")
                    glm2 = lh5.read("glm/det2", glm_file).view_as("ak")
                    glm = ak.Array({"det1": glm1, "det2": glm2})
                # glm should have the right evtid

                assert ak.all(glm.det1.evtid.to_numpy() == evtids)
                assert ak.all(glm.det2.evtid.to_numpy() == evtids)

                # total number of rows should be correct
                assert np.sum(glm.det1.n_rows) == len(evtids1_read)
                assert np.sum(glm.det2.n_rows) == len(evtids2_read)


def test_glm_iterator(tmptestdir):
    # make an glm

    # two files (no gaps and gaps)
    for test in ["simple", "gaps"]:
        stp_file = str(tmptestdir / f"{test}_test.lh5")
        glm_file = str(tmptestdir / f"{test}_glm.lh5")

        build_glm(
            stp_file,
            glm_file,
            id_name="evtid",
            evtid_buffer=1000,
            stp_buffer=100000,
        )

        # iterate over the glm and the test file
        # consider both a chunked and a full read
        # also try with the glm file being None
        for gt in [glm_file, None]:
            for n_rows in [5000, None]:
                for det in ["det1", "det2"]:
                    for reshaped in [True, False]:
                        evtids = None
                        glm_it = GLMIterator(
                            gt,
                            stp_file,
                            lh5_group=det,
                            start_row=0,
                            stp_field="stp",
                            n_rows=n_rows,
                            buffer=100,
                            reshaped_files=reshaped,
                        )
                        # get the overall evtids
                        for stps, _, _ in glm_it:
                            if stps is None:
                                continue

                            evtids = (
                                stps.view_as("ak").evtid
                                if evtids is None
                                else ak.concatenate((evtids, stps.view_as("ak").evtid))
                            )

                        evtids_read = lh5.read_as(
                            f"stp/{det}/evtid", str(tmptestdir / f"{test}_test.lh5"), "np"
                        )

                        assert ak.all(evtids == evtids_read)
