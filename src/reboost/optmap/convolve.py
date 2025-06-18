from __future__ import annotations

import logging
import re

import legendoptics.scintillate as sc
import numba
import numpy as np
import pint
from legendoptics import lar
from lgdo import lh5
from lgdo.lh5 import LH5Iterator
from lgdo.types import Array, Histogram, Table
from numba import njit, prange
from numpy.lib.recfunctions import structured_to_unstructured

from .numba_pdg import numba_pdgid_funcs

log = logging.getLogger(__name__)


OPTMAP_ANY_CH = -1
OPTMAP_SUM_CH = -2


def open_optmap(optmap_fn: str):
    maps = lh5.ls(optmap_fn)
    # only accept _<number> (/all is read separately)
    det_ntuples = [m for m in maps if re.match(r"_\d+$", m)]
    detids = np.array([int(m.lstrip("_")) for m in det_ntuples])
    detidx = np.arange(0, detids.shape[0])

    optmap_all = lh5.read("/all/p_det", optmap_fn)
    assert isinstance(optmap_all, Histogram)
    optmap_edges = tuple([b.edges for b in optmap_all.binning])

    ow = np.empty((detidx.shape[0] + 2, *optmap_all.weights.nda.shape), dtype=np.float64)
    # 0, ..., len(detidx)-1 AND OPTMAP_ANY_CH might be negative.
    ow[OPTMAP_ANY_CH] = optmap_all.weights.nda
    for i, nt in zip(detidx, det_ntuples):
        optmap = lh5.read(f"/{nt}/p_det", optmap_fn)
        assert isinstance(optmap, Histogram)
        ow[i] = optmap.weights.nda

    # if we have any individual channels registered, the sum is potentially larger than the
    # probability to find _any_ hit.
    if len(detidx) != 0:
        ow[OPTMAP_SUM_CH] = np.sum(ow[0:-2], axis=0, where=(ow[0:-2] >= 0))
        assert not np.any(ow[OPTMAP_SUM_CH] < 0)
    else:
        detidx = np.array([OPTMAP_ANY_CH])
        detids = np.array([0])
        ow[OPTMAP_SUM_CH] = ow[OPTMAP_ANY_CH]

    # give this check some numerical slack.
    if np.any(
        np.abs(
            ow[OPTMAP_SUM_CH][ow[OPTMAP_ANY_CH] >= 0] - ow[OPTMAP_ANY_CH][ow[OPTMAP_ANY_CH] >= 0]
        )
        < -1e-15
    ):
        msg = "optical map does not fulfill relation sum(p_i) >= p_any"
        raise ValueError(msg)

    try:
        # check the exponent from the optical map file
        optmap_multi_det_exp = lh5.read("/_hitcounts_exp", optmap_fn).value
        assert isinstance(optmap_multi_det_exp, float)
        if np.isfinite(optmap_multi_det_exp):
            msg = f"found finite _hitcounts_exp {optmap_multi_det_exp} which is not supported any more"
            raise RuntimeError(msg)
    except KeyError:  # the _hitcounts_exp might not be always present.
        pass

    return detids, detidx, optmap_edges, ow


def iterate_stepwise_depositions(
    edep_df: np.rec.recarray,
    optmap_for_convolve,
    scint_mat_params: sc.ComputedScintParams,
    rng: np.random.Generator = None,
    dist: str = "poisson",
    mode: str = "no-fano",
):
    # those np functions are not supported by numba, but needed for efficient array access below.
    if "xloc_pre" in edep_df.dtype.names:
        x0 = structured_to_unstructured(edep_df[["xloc_pre", "yloc_pre", "zloc_pre"]], np.float64)
        x1 = structured_to_unstructured(
            edep_df[["xloc_post", "yloc_post", "zloc_post"]], np.float64
        )
    else:
        x0 = structured_to_unstructured(edep_df[["xloc", "yloc", "zloc"]], np.float64)
        x1 = None

    rng = np.random.default_rng() if rng is None else rng
    output_map, res = _iterate_stepwise_depositions(
        edep_df, x0, x1, rng, *optmap_for_convolve, scint_mat_params, dist, mode
    )
    if res["any_no_stats"] > 0 or res["det_no_stats"] > 0:
        log.warning(
            "had edep out in voxels without stats: %d (%.2f%%)",
            res["any_no_stats"],
            res["det_no_stats"],
        )
    if res["oob"] > 0:
        log.warning(
            "had edep out of map bounds: %d (%.2f%%)",
            res["oob"],
            (res["oob"] / (res["ib"] + res["oob"])) * 100,
        )
    log.debug(
        "VUV_primary %d ->hits_any %d ->hits %d (%.2f %% primaries detected)",
        res["vuv_primary"],
        res["hits_any"],
        res["hits"],
        (res["hits_any"] / res["vuv_primary"]) * 100,
    )
    log.debug("hits/hits_any %.2f", res["hits"] / res["hits_any"])
    return output_map


_pdg_func = numba_pdgid_funcs()


@njit
def _pdgid_to_particle(pdgid: int) -> sc.ParticleIndex:
    abs_pdgid = abs(pdgid)
    if abs_pdgid == 1000020040:
        return sc.PARTICLE_INDEX_ALPHA
    if abs_pdgid == 1000010020:
        return sc.PARTICLE_INDEX_DEUTERON
    if abs_pdgid == 1000010030:
        return sc.PARTICLE_INDEX_TRITON
    if _pdg_func.is_nucleus(pdgid):
        return sc.PARTICLE_INDEX_ION
    return sc.PARTICLE_INDEX_ELECTRON


__counts_per_bin_key_type = numba.types.UniTuple(numba.types.int64, 3)


# - run with NUMBA_FULL_TRACEBACKS=1 NUMBA_BOUNDSCHECK=1 for testing/checking
# - cache=True does not work with outer prange, i.e. loading the cached file fails (numba bug?)
# - the output dictionary is not threadsafe, so parallel=True is not working with it.
@njit(parallel=False, nogil=True, cache=True)
def _iterate_stepwise_depositions(
    edep_df,
    x0,
    x1,
    rng,
    detids,
    detidx,
    optmap_edges,
    optmap_weights,
    scint_mat_params: sc.ComputedScintParams,
    dist: str,
    mode: str,
):
    pdgid_map = {}
    output_map = {}
    oob = ib = ph_cnt = ph_det = ph_det2 = any_no_stats = det_no_stats = 0  # for statistics
    for rowid in prange(edep_df.shape[0]):
        # if rowid % 100000 == 0:
        #     print(rowid)
        t = edep_df[rowid]

        # get the particle information.
        if t.particle not in pdgid_map:
            pdgid_map[t.particle] = (_pdgid_to_particle(t.particle), _pdg_func.charge(t.particle))

        # do the scintillation.
        part, charge = pdgid_map[t.particle]

        # if we have both pre and post step points use them
        # else pass as None

        scint_times = sc.scintillate(
            scint_mat_params,
            x0[rowid],
            x1[rowid] if x1 is not None else None,
            t.v_pre if x1 is not None else None,
            t.v_post if x1 is not None else None,
            t.time,
            part,
            charge,
            t.edep,
            rng,
            emission_term_model=("poisson" if mode == "no-fano" else "normal_fano"),
        )
        if scint_times.shape[0] == 0:  # short-circuit if we have no photons at all.
            continue
        ph_cnt += scint_times.shape[0]

        # coordinates -> bins of the optical map.
        bins = np.empty((scint_times.shape[0], 3), dtype=np.int64)
        for j in range(3):
            bins[:, j] = np.digitize(scint_times[:, j + 1], optmap_edges[j])
            # normalize all out-of-bounds bins just to one end.
            bins[:, j][bins[:, j] == optmap_edges[j].shape[0]] = 0

        # there are _much_ less unique bins, unfortunately np.unique(..., axis=n) does not work
        # with numba; also np.sort(..., axis=n) also does not work.

        counts_per_bin = numba.typed.Dict.empty(
            key_type=__counts_per_bin_key_type,
            value_type=np.int64,
        )

        # get probabilities from map.
        hitcount = np.zeros((detidx.shape[0], bins.shape[0]), dtype=np.int64)
        for j in prange(bins.shape[0]):
            # note: subtract 1 from bins, to account for np.digitize output.
            cur_bins = (bins[j, 0] - 1, bins[j, 1] - 1, bins[j, 2] - 1)
            if cur_bins[0] == -1 or cur_bins[1] == -1 or cur_bins[2] == -1:
                oob += 1
                continue  # out-of-bounds of optmap
            ib += 1

            px_any = optmap_weights[OPTMAP_ANY_CH, cur_bins[0], cur_bins[1], cur_bins[2]]
            if px_any < 0.0:
                any_no_stats += 1
                continue
            if px_any == 0.0:
                continue

            if dist == "multinomial":
                if rng.uniform() >= px_any:
                    continue
                ph_det += 1
                # we detect this energy deposition; we should at least get one photon out here!

                detsel_size = 1

                px_sum = optmap_weights[OPTMAP_SUM_CH, cur_bins[0], cur_bins[1], cur_bins[2]]
                assert px_sum >= 0.0  # should not be negative.
                detp = np.empty(detidx.shape, dtype=np.float64)
                had_det_no_stats = 0
                for d in detidx:
                    # normalize so that sum(detp) = 1
                    detp[d] = optmap_weights[d, cur_bins[0], cur_bins[1], cur_bins[2]] / px_sum
                    if detp[d] < 0.0:
                        had_det_no_stats = 1
                        detp[d] = 0.0
                det_no_stats += had_det_no_stats

                # should be equivalent to rng.choice(detidx, size=detsel_size, p=detp)
                detsel = detidx[
                    np.searchsorted(np.cumsum(detp), rng.random(size=(detsel_size,)), side="right")
                ]
                for d in detsel:
                    hitcount[d, j] += 1
                ph_det2 += detsel.shape[0]

            elif dist == "poisson":
                # store the photon count in each bin, to sample them all at once below.
                if cur_bins not in counts_per_bin:
                    counts_per_bin[cur_bins] = 1
                else:
                    counts_per_bin[cur_bins] += 1

            else:
                msg = "unknown distribution"
                raise RuntimeError(msg)

        if dist == "poisson":
            for j, (cur_bins, ph_counts_to_poisson) in enumerate(counts_per_bin.items()):
                had_det_no_stats = 0
                had_any = 0
                for d in detidx:
                    detp = optmap_weights[d, cur_bins[0], cur_bins[1], cur_bins[2]]
                    if detp < 0.0:
                        had_det_no_stats = 1
                        continue
                    pois_cnt = rng.poisson(lam=ph_counts_to_poisson * detp)
                    hitcount[d, j] += pois_cnt
                    ph_det2 += pois_cnt
                    had_any = 1
                ph_det += had_any
                det_no_stats += had_det_no_stats

        assert scint_times.shape[0] >= hitcount.shape[1]  # TODO: use the right assertion here.
        out_hits_len = np.sum(hitcount)
        if out_hits_len > 0:
            out_times = np.empty(out_hits_len, dtype=np.float64)
            out_det = np.empty(out_hits_len, dtype=np.int64)
            out_idx = 0
            for d in detidx:
                hc_d_plane_max = np.max(hitcount[d, :])
                # untangle the hitcount array in "planes" that only contain the given number of hits per
                # channel. example: assume a "histogram" of hits per channel:
                #     x |   |    <-- this is plane 2 with 1 hit ("max plane")
                #     x |   | x  <-- this is plane 1 with 2 hits
                # ch: 1 | 2 | 3
                for hc_d_plane_cnt in range(1, hc_d_plane_max + 1):
                    hc_d_plane = hitcount[d, :] >= hc_d_plane_cnt
                    hc_d_plane_len = np.sum(hc_d_plane)
                    if hc_d_plane_len == 0:
                        continue

                    # note: we assume "immediate" propagation after scintillation. Here, a single timestamp
                    # might be coipied to output/"detected" twice.
                    out_times[out_idx : out_idx + hc_d_plane_len] = scint_times[hc_d_plane, 0]
                    out_det[out_idx : out_idx + hc_d_plane_len] = detids[d]
                    out_idx += hc_d_plane_len
            assert out_idx == out_hits_len  # ensure that all of out_{det,times} is filled.
            output_map[np.int64(rowid)] = (t.evtid, out_det, out_times)

    stats = {
        "oob": oob,
        "ib": ib,
        "vuv_primary": ph_cnt,
        "hits_any": ph_det,
        "hits": ph_det2,
        "any_no_stats": any_no_stats,
        "det_no_stats": det_no_stats,
    }
    return output_map, stats


def get_output_table(output_map):
    ph_count_o = 0
    for _rawid, (_evtid, det, _times) in output_map.items():
        ph_count_o += det.shape[0]

    out_idx = 0
    out_evtid = np.empty(ph_count_o, dtype=np.int64)
    out_det = np.empty(ph_count_o, dtype=np.int64)
    out_times = np.empty(ph_count_o, dtype=np.float64)
    for _rawid, (evtid, det, times) in output_map.items():
        o_len = det.shape[0]
        out_evtid[out_idx : out_idx + o_len] = evtid
        out_det[out_idx : out_idx + o_len] = det
        out_times[out_idx : out_idx + o_len] = times
        out_idx += o_len

    tbl = Table({"evtid": Array(out_evtid), "det_uid": Array(out_det), "time": Array(out_times)})
    return ph_count_o, tbl


def convolve(
    map_file: str,
    edep_file: str,
    edep_path: str,
    material: str,
    output_file: str | None = None,
    buffer_len: int = int(1e6),
    dist_mode: str = "poisson+no-fano",
):
    if material not in ["lar", "pen"]:
        msg = f"unknown material {material} for scintillation"
        raise ValueError(msg)

    if material == "lar":
        scint_mat_params = sc.precompute_scintillation_params(
            lar.lar_scintillation_params(),
            lar.lar_lifetimes().as_tuple(),
        )
    elif material == "pen":
        scint_mat_params = sc.precompute_scintillation_params(
            lar.pen_scintillation_params(),
            (1 * pint.get_application_registry().ns),  # dummy!
        )

    # special handling of distributions and flags.
    dist, mode = dist_mode.split("+")
    if (
        dist not in ("multinomial", "poisson")
        or mode not in ("", "no-fano")
        or (dist == "poisson" and mode != "no-fano")
    ):
        msg = f"unsupported statistical distribution {dist_mode} for scintillation emission"
        raise ValueError(msg)

    log.info("opening map %s", map_file)
    optmap_for_convolve = open_optmap(map_file)

    log.info("opening energy deposition hit output %s", edep_file)
    it = LH5Iterator(edep_file, edep_path, buffer_len=buffer_len)

    for it_count, edep_lgdo in enumerate(it):
        edep_df = edep_lgdo.view_as("pd").to_records()

        log.info("start event processing (%d)", it_count)
        output_map = iterate_stepwise_depositions(
            edep_df, optmap_for_convolve, scint_mat_params, dist=dist, mode=mode
        )

        log.info("store output photon hits (%d)", it_count)
        ph_count_o, tbl = get_output_table(output_map)
        log.debug(
            "output photons: %d energy depositions -> %d photons", len(output_map), ph_count_o
        )
        if output_file is not None:
            lh5.write(tbl, "optical", lh5_file=output_file, group="stp", wo_mode="append")
