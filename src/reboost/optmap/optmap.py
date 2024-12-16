from __future__ import annotations

import contextlib
import ctypes
import logging
import math
import multiprocessing as mp
from collections.abc import Mapping

import numpy as np
from lgdo import Histogram, lh5
from numpy.typing import NDArray

log = logging.getLogger(__name__)


def _fill_histogram(h: NDArray, binning: NDArray, xyz: NDArray) -> None:
    assert xyz.shape[1] == 3
    xyz = xyz.T

    idx = np.zeros(xyz.shape[1], np.int64)  # bin indices for flattened array
    oor_mask = np.ones(xyz.shape[1], np.bool_)  # mask to remove out of range values
    stride = [s // h.dtype.itemsize for s in h.strides]
    dims = range(xyz.shape[0])
    for col, ax, s, dim in zip(xyz, binning, stride, dims):
        assert ax.is_range
        assert ax.closedleft
        oor_mask &= (ax.first <= col) & (col < ax.last)
        idx_s = np.floor((col - ax.first).astype(np.float64) / ax.step).astype(np.int64)
        assert np.all(idx_s[oor_mask] < h.shape[dim])
        idx += s * idx_s

    # increment bin contents
    idx = idx[oor_mask]
    np.add.at(h.reshape(-1), idx, 1)


class OpticalMap:
    def __init__(self, name: str, settings: Mapping[str, str], use_shmem: bool = False):
        self.settings = settings
        self.name = name
        self.use_shmem = use_shmem

        self.h_vertex = None
        self.h_hits = None
        self.h_prob = None
        self.h_prob_uncert = None

        self.binning = None

        if settings is not None:
            self._single_shape = tuple(self.settings["bins"])

            binedge_attrs = {"units": "m"}
            bins = self.settings["bins"]
            bounds = self.settings["range_in_m"]
            self.binning = [
                Histogram.Axis(
                    None,
                    bounds[i][0],
                    bounds[i][1],
                    (bounds[i][1] - bounds[i][0]) / bins[i],
                    True,
                    binedge_attrs,
                )
                for i in range(3)
            ]

    def create_empty(name: str, settings: Mapping[str, str]) -> OpticalMap:
        om = OpticalMap(name, settings)
        om.h_vertex = om._prepare_hist()
        om.h_hits = om._prepare_hist()
        return om

    @staticmethod
    def load_from_file(lh5_file: str, group: str) -> OpticalMap:
        om = OpticalMap(group, None)

        def read_hist(name: str, fn: str, group: str = "all"):
            h = lh5.read(f"/{group}/{name}", lh5_file=fn)
            if not isinstance(h, Histogram):
                msg = f"encountered invalid optical map while reading /{group}/{name} in {fn}"
                raise ValueError(msg)
            return h.weights.nda, h.binning

        om.h_vertex, om.binning = read_hist("nr_gen", lh5_file, group=group)
        om.h_hits, _ = read_hist("nr_det", lh5_file, group=group)
        om.h_prob, _ = read_hist("p_det", lh5_file, group=group)
        om.h_prob_uncert, _ = read_hist("p_det_err", lh5_file, group=group)
        return om

    def _prepare_hist(self) -> np.ndarray:
        """Prepare an empty histogram with the parameters global to this map instance."""
        if self.use_shmem:
            assert mp.current_process().name == "MainProcess"
            a = self._mp_man.Array(ctypes.c_double, math.prod(self._single_shape))
            self._nda(a).fill(0)
            return a
        return np.zeros(shape=self._single_shape, dtype=np.float64)

    def _nda(self, h) -> NDArray:
        if not self.use_shmem:
            return h
        return np.ndarray(self._single_shape, dtype=np.float64, buffer=h.get_obj())

    def _lock_nda(self, h):
        if not self.use_shmem:
            return contextlib.nullcontext
        return h.get_lock

    def _mp_preinit(self, mp_man, vertex: bool) -> None:
        self._mp_man = mp_man
        if self.h_vertex is None and vertex:
            self.h_vertex = self._prepare_hist()
        if self.h_hits is None:
            self.h_hits = self._prepare_hist()

    def fill_vertex(self, loc) -> None:
        if self.h_vertex is None:
            self.h_vertex = self._prepare_hist()
        with self._lock_nda(self.h_vertex)():
            _fill_histogram(self._nda(self.h_vertex), self.binning, loc)

    def fill_hits(self, loc) -> None:
        if self.h_hits is None:
            self.h_hits = self._prepare_hist()
        with self._lock_nda(self.h_hits)():
            _fill_histogram(self._nda(self.h_hits), self.binning, loc)

    def _divide_hist(self, h1: NDArray, h2: NDArray) -> tuple[NDArray, NDArray]:
        """Calculate the ratio (and its standard error) from two histograms."""

        h1 = self._nda(h1)
        h2 = self._nda(h2)

        ratio_0 = self._prepare_hist()
        ratio_err_0 = self._prepare_hist()
        ratio, ratio_err = self._nda(ratio_0), self._nda(ratio_err_0)

        ratio[:] = np.divide(h1, h2, where=(h2 != 0))
        ratio[h2 == 0] = -1  # -1 denotes no statistics.

        if np.any(ratio > 1):
            msg = "encountered cell(s) with more hits than primaries"
            raise RuntimeError(msg)

        # compute uncertainty according to Bernoulli statistics.
        # TODO: this does not make sense for ratio==1
        ratio_err[h2 != 0] = np.sqrt((ratio[h2 != 0]) * (1 - ratio[h2 != 0]) / h2[h2 != 0])
        ratio_err[h2 == 0] = -1  # -1 denotes no statistics.

        return ratio_0, ratio_err_0

    def create_probability(self) -> None:
        self.h_prob, self.h_prob_uncert = self._divide_hist(self.h_hits, self.h_vertex)

    def write_lh5(self, lh5_file: str, group: str = "all") -> None:
        def write_hist(h: NDArray, name: str, fn: str, group: str = "all"):
            lh5.write(
                Histogram(self._nda(h), self.binning),
                name,
                fn,
                group=group,
                wo_mode="write_safe",
            )

        write_hist(self.h_vertex, "nr_gen", lh5_file, group=group)
        write_hist(self.h_hits, "nr_det", lh5_file, group=group)
        write_hist(self.h_prob, "p_det", lh5_file, group=group)
        write_hist(self.h_prob_uncert, "p_det_err", lh5_file, group=group)

    def check_histograms(self, include_prefix: bool = False) -> None:
        log_prefix = "" if not include_prefix else self.name + " - "

        def _warn(fmt: str, *args):
            log.warning("%s" + fmt, log_prefix, *args)  # noqa: G003

        h_vertex = self._nda(self.h_vertex)
        h_prob = self._nda(self.h_prob)
        h_prob_uncert = self._nda(self.h_prob_uncert)

        ncells = h_vertex.shape[0] * h_vertex.shape[1] * h_vertex.shape[2]

        missing_v = np.sum(h_vertex <= 0)  # bins without vertices.
        if missing_v > 0:
            _warn("%d missing_v %.2f %%", missing_v, missing_v / ncells * 100)

        missing_p = np.sum(h_prob <= 0)  # bins without hist.
        if missing_p > 0:
            _warn("%d missing_p %.2f %%", missing_p, missing_p / ncells * 100)

        non_phys = np.sum(h_prob > 1)  # non-physical events with probability > 1.
        if non_phys > 0:
            _warn(
                "%d voxels (%.2f %%) with non-physical probability (p>1)",
                non_phys,
                non_phys / ncells * 100,
            )

        # warnings on insufficient statistics.
        large_error = np.sum(h_prob_uncert > 0.01 * h_prob)
        if large_error > 0:
            _warn(
                "%d voxels (%.2f %%) with large relative statistical uncertainty (> 1 %%)",
                large_error,
                large_error / ncells * 100,
            )

        primaries_low_stats_th = 100
        low_stat_zero = np.sum((h_vertex < primaries_low_stats_th) & (h_prob == 0))
        if low_stat_zero > 0:
            _warn(
                "%d voxels (%.2f %%) with non reliable probability estimate (p=0 and primaries < %d)",
                low_stat_zero,
                low_stat_zero / ncells * 100,
                primaries_low_stats_th,
            )
        low_stat_one = np.sum((h_vertex < primaries_low_stats_th) & (h_prob == 1))
        if low_stat_one > 0:
            _warn(
                "%d voxels (%.2f %%) with non reliable probability estimate (p=1 and primaries < %d)",
                low_stat_one,
                low_stat_one / ncells * 100,
                primaries_low_stats_th,
            )