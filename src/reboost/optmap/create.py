from __future__ import annotations

import copy
import gc
import logging
import multiprocessing as mp
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from lgdo import Histogram, lh5
from numba import njit
from numpy.typing import NDArray

from ..log_utils import setup_log
from .evt import (
    generate_optmap_evt,
    get_optical_detectors_from_geom,
)
from .optmap import OpticalMap

log = logging.getLogger(__name__)


def _optmaps_for_channels(
    all_det_ids: dict[int, str],
    settings,
    chfilter: tuple[str | int] | Literal["*"] = (),
    use_shmem: bool = False,
):
    if chfilter != "*":
        chfilter = [str(ch) for ch in chfilter]  # normalize types
        optmap_det_ids = {det: name for det, name in all_det_ids.items() if str(det) in chfilter}
    else:
        optmap_det_ids = all_det_ids

    log.info("creating empty optmaps")
    optmap_count = len(optmap_det_ids) + 1
    optmaps = [
        OpticalMap("all" if i == 0 else list(optmap_det_ids.values())[i - 1], settings, use_shmem)
        for i in range(optmap_count)
    ]

    return all_det_ids, optmaps, optmap_det_ids


@njit(cache=True)
def _compute_hit_maps(hitcounts, optmap_count, ch_idx_to_optmap):
    mask = np.zeros((hitcounts.shape[0], optmap_count), dtype=np.bool_)
    counts = hitcounts.sum(axis=1)
    for idx in range(hitcounts.shape[0]):
        if counts[idx] == 0:
            continue

        for ch_idx in range(hitcounts.shape[1]):
            c = hitcounts[idx, ch_idx]
            if c > 0:  # detected
                mask[idx, 0] = True
                mask_idx = ch_idx_to_optmap[ch_idx]
                if mask_idx > 0:
                    mask[idx, mask_idx] = True
    return mask


def _fill_hit_maps(optmaps: list[OpticalMap], loc, hitcounts: NDArray, ch_idx_to_map_idx):
    masks = _compute_hit_maps(hitcounts, len(optmaps), ch_idx_to_map_idx)

    for i in range(len(optmaps)):
        locm = loc[masks[:, i]]
        optmaps[i].fill_hits(locm)


def _create_optical_maps_process_init(optmaps, log_level) -> None:
    # need to use shared global state. passing the shared memory arrays via "normal" arguments to
    # the worker function is not supported...
    global _shared_optmaps  # noqa: PLW0603
    _shared_optmaps = optmaps

    # setup logging in the worker process.
    setup_log(log_level, multiproc=True)


def _create_optical_maps_process(
    optmap_events_fn, buffer_len, all_det_ids, ch_idx_to_map_idx
) -> bool:
    log.info("started worker task for %s", optmap_events_fn)
    x = _create_optical_maps_chunk(
        optmap_events_fn,
        buffer_len,
        all_det_ids,
        _shared_optmaps,
        ch_idx_to_map_idx,
    )
    log.info("finished worker task for %s", optmap_events_fn)
    return x


def _create_optical_maps_chunk(
    optmap_events_fn, buffer_len, all_det_ids, optmaps, ch_idx_to_map_idx
) -> bool:
    cols = [str(c) for c in all_det_ids]
    optmap_events_it = generate_optmap_evt(optmap_events_fn, cols, buffer_len)

    for it_count, events_lgdo in enumerate(optmap_events_it):
        optmap_events = events_lgdo.view_as("pd")
        hitcounts = optmap_events[cols].to_numpy()
        loc = optmap_events[["xloc", "yloc", "zloc"]].to_numpy()

        log.debug("filling vertex histogram (%d)", it_count)
        optmaps[0].fill_vertex(loc)

        log.debug("filling hits histogram (%d)", it_count)
        _fill_hit_maps(optmaps, loc, hitcounts, ch_idx_to_map_idx)

    # commit the final part of the hits to the maps.
    for i in range(len(optmaps)):
        optmaps[i].fill_hits_flush()
        gc.collect()

    return True


def create_optical_maps(
    optmap_events_fn: list[str],
    settings,
    buffer_len: int = int(5e6),
    chfilter: tuple[str | int] | Literal["*"] = (),
    output_lh5_fn: str | None = None,
    after_save: Callable[[int, str, OpticalMap]] | None = None,
    check_after_create: bool = False,
    n_procs: int | None = 1,
    geom_fn: str | None = None,
) -> None:
    """Create optical maps.

    Parameters
    ----------
    optmap_events_fn
        list of filenames to lh5 files, that can either be stp files from remage or "optmap-evt"
        files with a table ``/optmap_evt`` with columns ``{x,y,z}loc`` and one column (with numeric
        header) for each SiPM channel.
    chfilter
        tuple of detector ids that will be included in the resulting optmap. Those have to match
        the column names in ``optmap_events_fn``.
    n_procs
        number of processors, ``1`` for sequential mode, or ``None`` to use all processors.
    """
    if len(optmap_events_fn) == 0:
        msg = "no input files specified"
        raise ValueError(msg)

    use_shmem = n_procs is None or n_procs > 1

    optmap_evt_columns = get_optical_detectors_from_geom(geom_fn)

    all_det_ids, optmaps, optmap_det_ids = _optmaps_for_channels(
        optmap_evt_columns, settings, chfilter=chfilter, use_shmem=use_shmem
    )

    # indices for later use in _compute_hit_maps.
    ch_idx_to_map_idx = np.array(
        [
            list(optmap_det_ids.keys()).index(d) + 1 if d in optmap_det_ids else -1
            for d in all_det_ids
        ]
    )
    assert np.sum(ch_idx_to_map_idx > 0) == len(optmaps) - 1

    log.info(
        "creating optical map groups: %s",
        ", ".join(["all", *[str(t) for t in optmap_det_ids.items()]]),
    )

    q = []

    # sequential mode.
    if not use_shmem:
        for fn in optmap_events_fn:
            q.append(
                _create_optical_maps_chunk(fn, buffer_len, all_det_ids, optmaps, ch_idx_to_map_idx)
            )
    else:
        ctx = mp.get_context("forkserver")
        for i in range(len(optmaps)):
            optmaps[i]._mp_preinit(ctx, vertex=(i == 0))

        # note: errors thrown in initializer will make the main process hang in an endless loop.
        # unfortunately, we cannot pass the objects later, as they contain shmem/array handles.
        pool = ctx.Pool(
            n_procs,
            initializer=_create_optical_maps_process_init,
            initargs=(optmaps, log.getEffectiveLevel()),
            maxtasksperchild=1,  # re-create worker after each task, to avoid leaking memory.
        )

        pool_results = []
        for fn in optmap_events_fn:
            r = pool.apply_async(
                _create_optical_maps_process,
                args=(fn, buffer_len, all_det_ids, ch_idx_to_map_idx),
            )
            pool_results.append((r, fn))

        pool.close()
        for r, fn in pool_results:
            try:
                q.append(r.get())
            except BaseException as e:
                msg = f"error while processing file {fn}"
                raise RuntimeError(msg) from e  # re-throw errors of workers.
        log.debug("got all worker results")
        pool.join()
        log.info("joined worker process pool")

    if len(q) != len(optmap_events_fn):
        log.error("got %d results for %d files", len(q), len(optmap_events_fn))

    # all maps share the same vertex histogram.
    for i in range(1, len(optmaps)):
        optmaps[i].h_vertex = optmaps[0].h_vertex

    log.info("computing probability and storing to %s", output_lh5_fn)
    for i in range(len(optmaps)):
        optmaps[i].create_probability()
        if check_after_create:
            optmaps[i].check_histograms()
        group = "all" if i == 0 else "channels/" + list(optmap_det_ids.values())[i - 1]
        if output_lh5_fn is not None:
            optmaps[i].write_lh5(lh5_file=output_lh5_fn, group=group)

        if after_save is not None:
            after_save(i, group, optmaps[i])

        optmaps[i] = None  # clear some memory.


def list_optical_maps(lh5_file: str) -> list[str]:
    maps = list(lh5.ls(lh5_file, "/channels/"))
    if "all" in lh5.ls(lh5_file):
        maps.append("all")
    return maps


def _merge_optical_maps_process(
    d: str,
    map_l5_files: list[str],
    output_lh5_fn: str,
    settings,
    check_after_create: bool = False,
    write_part_file: bool = False,
) -> bool:
    log.info("merging optical map group: %s", d)
    merged_map = OpticalMap.create_empty(d, settings)
    merged_nr_gen = merged_map.h_vertex
    merged_nr_det = merged_map.h_hits

    all_edges = None
    for optmap_fn in map_l5_files:
        nr_det = lh5.read(f"/{d}/_nr_det", optmap_fn)
        assert isinstance(nr_det, Histogram)
        nr_gen = lh5.read(f"/{d}/_nr_gen", optmap_fn)
        assert isinstance(nr_gen, Histogram)

        assert OpticalMap._edges_eq(nr_det.binning, nr_gen.binning)
        if all_edges is not None and not OpticalMap._edges_eq(nr_det.binning, all_edges):
            msg = "edges of input optical maps differ"
            raise ValueError(msg)
        all_edges = nr_det.binning

        # now that we validated that the map dimensions are equal, add up the actual data (in counts).
        merged_nr_det += nr_det.weights.nda
        merged_nr_gen += nr_gen.weights.nda

    merged_map.create_probability()
    if check_after_create:
        merged_map.check_histograms(include_prefix=True)

    if write_part_file:
        d_for_tmp = d.replace("/", "_")
        output_lh5_fn = f"{output_lh5_fn}_{d_for_tmp}.mappart.lh5"
    wo_mode = "overwrite_file" if write_part_file else "write_safe"
    merged_map.write_lh5(lh5_file=output_lh5_fn, group=d, wo_mode=wo_mode)

    return output_lh5_fn


def merge_optical_maps(
    map_l5_files: list[str],
    output_lh5_fn: str,
    settings,
    check_after_create: bool = False,
    n_procs: int | None = 1,
) -> None:
    """Merge optical maps from multiple files.

    Parameters
    ----------
    n_procs
        number of processors, ``1`` for sequential mode, or ``None`` to use all processors.
    """
    # verify that we have the same maps in all files.
    all_det_ntuples = None
    for optmap_fn in map_l5_files:
        det_ntuples = list_optical_maps(optmap_fn)
        if all_det_ntuples is not None and det_ntuples != all_det_ntuples:
            msg = "available optical maps in input files differ"
            raise ValueError(msg)
        all_det_ntuples = det_ntuples

    log.info("merging optical map groups: %s", ", ".join(all_det_ntuples))

    use_mp = (n_procs is None or n_procs > 1) and len(all_det_ntuples) > 1

    if not use_mp:
        # sequential mode: merge maps one-by-one.
        for d in all_det_ntuples:
            _merge_optical_maps_process(
                d, map_l5_files, output_lh5_fn, settings, check_after_create, use_mp
            )
    else:
        ctx = mp.get_context("forkserver")

        # note: errors thrown in initializer will make the main process hang in an endless loop.
        pool = ctx.Pool(
            n_procs,
            initializer=_create_optical_maps_process_init,
            initargs=(None, log.getEffectiveLevel()),
            maxtasksperchild=1,  # re-create worker after each task, to avoid leaking memory.
        )

        pool_results = []

        # merge maps in workers.
        for d in all_det_ntuples:
            r = pool.apply_async(
                _merge_optical_maps_process,
                args=(d, map_l5_files, output_lh5_fn, settings, check_after_create, use_mp),
            )
            pool_results.append((r, d))

        pool.close()
        q = []
        for r, d in pool_results:
            try:
                q.append((d, r.get()))
            except BaseException as e:
                msg = f"error while processing map {d}"
                raise RuntimeError(msg) from e  # re-throw errors of workers.

        log.debug("got all worker results")
        pool.join()
        log.info("joined worker process pool")

        # transfer to actual output file.
        for d, part_fn in q:
            assert isinstance(part_fn, str)
            for h_name in ("_nr_det", "_nr_gen", "prob", "prob_unc"):
                obj = f"/{d}/{h_name}"
                log.info("transfer %s from %s", obj, part_fn)
                h = lh5.read(obj, part_fn)
                assert isinstance(h, Histogram)
                lh5.write(h, obj, output_lh5_fn, wo_mode="write_safe")
            Path(part_fn).unlink()


def check_optical_map(map_l5_file: str):
    """Run a health check on the map file.

    This checks for consistency, and outputs details on map statistics.
    """
    if (
        "_hitcounts_exp" in lh5.ls(map_l5_file)
        and lh5.read("_hitcounts_exp", lh5_file=map_l5_file).value != np.inf
    ):
        log.error("unexpected _hitcounts_exp not equal to positive infinity")
        return

    all_binning = None
    for submap in list_optical_maps(map_l5_file):
        try:
            om = OpticalMap.load_from_file(map_l5_file, submap)
        except Exception:
            log.exception("error while loading optical map %s", submap)
            continue
        om.check_histograms(include_prefix=True)

        if all_binning is not None and not OpticalMap._edges_eq(om.binning, all_binning):
            log.error("edges of optical map %s differ", submap)
        else:
            all_binning = om.binning


def rebin_optical_maps(map_l5_file: str, output_lh5_file: str, factor: int):
    """Rebin the optical map by an integral factor.

    .. note ::

        the factor has to divide the bincounts on all axes.
    """
    if not isinstance(factor, int) or factor <= 1:
        msg = f"invalid rebin factor {factor}"
        raise ValueError(msg)

    def _rebin_map(large: NDArray, factor: int) -> NDArray:
        factor = np.full(3, factor, dtype=int)
        sh = np.column_stack([np.array(large.shape) // factor, factor]).ravel()
        return large.reshape(sh).sum(axis=(1, 3, 5))

    for submap in list_optical_maps(map_l5_file):
        log.info("rebinning optical map group: %s", submap)

        om = OpticalMap.load_from_file(map_l5_file, submap)

        settings = om.get_settings()
        if not all(b % factor == 0 for b in settings["bins"]):
            msg = f"invalid factor {factor}, not a divisor"
            raise ValueError(msg)
        settings = copy.copy(settings)
        settings["bins"] = [b // factor for b in settings["bins"]]

        om_new = OpticalMap.create_empty(om.name, settings)
        om_new.h_vertex = _rebin_map(om.h_vertex, factor)
        om_new.h_hits = _rebin_map(om.h_hits, factor)
        om_new.create_probability()
        om_new.write_lh5(lh5_file=output_lh5_file, group=submap, wo_mode="write_safe")
