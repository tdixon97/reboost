from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from lgdo import Array, Table, lh5

from reboost.optmap.create import (
    check_optical_map,
    create_optical_maps,
    list_optical_maps,
    merge_optical_maps,
    rebin_optical_maps,
)
from reboost.optmap.evt import build_optmap_evt
from reboost.optmap.optmap import OpticalMap


@pytest.fixture
def tbl_hits(tmptestdir):
    evt_count = 100
    rng = np.random.default_rng(1234)
    loc = rng.uniform(size=(evt_count, 3))
    evtids = np.arange(1, evt_count + 1)

    tbl_vertices = Table(
        {
            "evtid": Array(evtids),
            "xloc": Array(loc[:, 0]),
            "yloc": Array(loc[:, 1]),
            "zloc": Array(loc[:, 2]),
            "n_part": Array(np.ones(evt_count)),
            "time": Array(np.ones(evt_count)),
        }
    )

    mask = rng.uniform(size=evt_count) < 0.2
    hit_count = np.sum(mask)

    tbl_optical = Table(
        {
            "evtid": Array(evtids[mask]),
            "det_uid": Array(np.ones(hit_count, dtype=np.int_)),
            "wavelength": Array(rng.normal(loc=400, scale=30, size=hit_count)),
            "time": Array(2 * np.ones(hit_count)),
        }
    )

    hit_file = tmptestdir / "hit.lh5"
    lh5.write(tbl_vertices, name="vtx", lh5_file=hit_file, wo_mode="overwrite_file")
    lh5.write(tbl_optical, name="stp/optical", lh5_file=hit_file, wo_mode="overwrite")
    return (str(hit_file),)


def test_optmap_evt(tbl_hits, tmptestdir):
    evt_out_file = tmptestdir / "evt-out.lh5"
    build_optmap_evt(
        tbl_hits[0],
        str(evt_out_file),
        detectors=("1", "002", "003"),
        buffer_len=20,  # note: shorter window sizes (e.g. 10) do not work.
    )


@pytest.fixture
def tbl_evt_fns(tmptestdir) -> tuple[str]:
    evt_count = 100
    rng = np.random.default_rng(1234)
    loc = rng.uniform(size=(evt_count, 3))
    # hits = rng.geometric(p=0.9, size=(evt_count, 3)) - 1
    hits = rng.choice([1, 2, 3], size=evt_count)

    tbl_evt = Table(
        {
            "xloc": Array(loc[:, 0]),
            "yloc": Array(loc[:, 1]),
            "zloc": Array(loc[:, 2]),
            "1": Array((hits == 1).astype(int)),
            "2": Array((hits == 2).astype(int)),
            "3": Array((hits == 3).astype(int)),
        }
    )

    evt_file = tmptestdir / "evt.lh5"
    lh5.write(tbl_evt, name="optmap_evt", lh5_file=evt_file, wo_mode="overwrite_file")
    return (str(evt_file),)


@pytest.mark.filterwarnings("ignore::scipy.optimize._optimize.OptimizeWarning")
def test_optmap_create(tbl_hits, tmptestdir):
    settings = {
        "range_in_m": [[0, 1], [0, 1], [0, 1]],
        "bins": [10, 10, 10],
    }

    # test creation only with the summary map.
    map_fn = str(tmptestdir / "create-map-1.lh5")
    create_optical_maps(
        tbl_hits,
        settings,
        chfilter=(),
        output_lh5_fn=map_fn,
        geom_fn=f"{Path(__file__).parent}/test_optmap_dets.gdml",
    )
    assert list_optical_maps(map_fn) == ["all"]

    # test creation with all detectors.
    map_fn = str(tmptestdir / "create-map-2.lh5")
    create_optical_maps(
        tbl_hits,
        settings,
        chfilter=("1", "2", "3"),
        output_lh5_fn=map_fn,
        geom_fn=f"{Path(__file__).parent}/test_optmap_dets.gdml",
    )
    assert list_optical_maps(map_fn) == ["channels/S01", "channels/S02", "channels/S03", "all"]

    # test creation with some detectors.
    map_fn = str(tmptestdir / "create-map-3.lh5")
    create_optical_maps(
        tbl_hits,
        settings,
        chfilter=("1"),
        output_lh5_fn=map_fn,
        geom_fn=f"{Path(__file__).parent}/test_optmap_dets.gdml",
    )
    assert list_optical_maps(map_fn) == ["channels/S01", "all"]

    # test creation on multiple cores.
    map_fn = str(tmptestdir / "create-map-4.lh5")
    create_optical_maps(
        tbl_hits,
        settings,
        chfilter=("1", "2", "3"),
        output_lh5_fn=map_fn,
        n_procs=2,
        geom_fn=f"{Path(__file__).parent}/test_optmap_dets.gdml",
    )
    assert list_optical_maps(map_fn) == ["channels/S01", "channels/S02", "channels/S03", "all"]


@pytest.mark.filterwarnings("ignore::scipy.optimize._optimize.OptimizeWarning")
def test_optmap_merge(tbl_hits, tmptestdir):
    settings = {
        "range_in_m": [[0, 1], [0, 1], [0, 1]],
        "bins": [10, 10, 10],
    }

    map1_fn = str(tmptestdir / "merge-map1.lh5")
    create_optical_maps(
        tbl_hits,
        settings,
        chfilter=("1", "2", "3"),
        output_lh5_fn=map1_fn,
        geom_fn=f"{Path(__file__).parent}/test_optmap_dets.gdml",
    )
    map2_fn = str(tmptestdir / "merge-map2.lh5")
    create_optical_maps(
        tbl_hits,
        settings,
        chfilter=("1", "2", "3"),
        output_lh5_fn=map2_fn,
        geom_fn=f"{Path(__file__).parent}/test_optmap_dets.gdml",
    )

    # test in sequential mode.
    map_merged_fn = str(tmptestdir / "map-merged.lh5")
    merge_optical_maps([map1_fn, map2_fn], map_merged_fn, settings)
    assert list_optical_maps(map_merged_fn) == [
        "channels/S01",
        "channels/S02",
        "channels/S03",
        "all",
    ]

    # also test on multiple cores.
    map_merged_fn = str(tmptestdir / "map-merged-mp.lh5")
    merge_optical_maps([map1_fn, map2_fn], map_merged_fn, settings, n_procs=2)
    assert list_optical_maps(map_merged_fn) == [
        "channels/S01",
        "channels/S02",
        "channels/S03",
        "all",
    ]


@pytest.mark.filterwarnings("ignore::scipy.optimize._optimize.OptimizeWarning")
def test_optmap_rebin(tbl_hits, tmptestdir):
    settings = {
        "range_in_m": [[0, 1], [0, 1], [0, 1]],
        "bins": [10, 10, 10],
    }

    map1_fn = str(tmptestdir / "map-to-rebin.lh5")
    create_optical_maps(
        tbl_hits,
        settings,
        chfilter=("1", "2", "3"),
        output_lh5_fn=map1_fn,
        geom_fn=f"{Path(__file__).parent}/test_optmap_dets.gdml",
    )

    map_rebinned_fn = str(tmptestdir / "map-rebinned.lh5")
    rebin_optical_maps(map1_fn, map_rebinned_fn, factor=2)


@pytest.fixture
def tbl_edep(tmptestdir):
    evt_count = 20
    rng = np.random.default_rng(1234)
    edep_counts = rng.uniform(2, 8, size=evt_count).astype(np.int_)
    edep_count = np.sum(edep_counts)
    loc = rng.uniform(size=(edep_count, 6))

    evtids = np.arange(1, evt_count + 1)

    tbl_edep = Table(
        {
            "evtid": Array(np.repeat(evtids, edep_counts)),
            "particle": Array(22 * np.ones(edep_count, dtype=np.int64)),
            "edep": Array(rng.normal(loc=200, scale=2, size=edep_count)),
            "xloc_pre": Array(loc[:, 0]),
            "yloc_pre": Array(loc[:, 1]),
            "zloc_pre": Array(loc[:, 2]),
            "xloc_post": Array(loc[:, 3]),
            "yloc_post": Array(loc[:, 4]),
            "zloc_post": Array(loc[:, 5]),
            "time": Array(np.ones(edep_count)),
            "v_pre": Array(60 * np.ones(edep_count)),
            "v_post": Array(58 * np.ones(edep_count)),
        }
    )

    evt_file = tmptestdir / "edep.lh5"
    lh5.write(tbl_edep, name="/stp/x", lh5_file=evt_file, wo_mode="overwrite_file")
    return evt_file


def test_optmap_save_and_load(tmptestdir, tbl_hits):
    settings = {
        "range_in_m": [[0, 1], [0, 1], [0, 1]],
        "bins": [10, 10, 10],
    }

    map_fn = str(tmptestdir / "map-load.lh5")
    create_optical_maps(
        tbl_hits,
        settings,
        chfilter=("1", "2", "3"),
        output_lh5_fn=map_fn,
        geom_fn=f"{Path(__file__).parent}/test_optmap_dets.gdml",
    )

    assert list_optical_maps(map_fn) == ["channels/S01", "channels/S02", "channels/S03", "all"]
    om = OpticalMap.load_from_file(map_fn, "all")
    assert isinstance(om, OpticalMap)
    om = OpticalMap.load_from_file(map_fn, "all")
    assert isinstance(om, OpticalMap)

    check_optical_map(map_fn)
