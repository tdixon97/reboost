from __future__ import annotations

import awkward as ak
import numpy as np


def def_chain(funcs, kwargs_list):
    def func(data):
        tmp = data
        for f, kw in zip(funcs, kwargs_list):
            tmp = f(tmp, **kw)

        return tmp

    return func


def group_by_evtid(data):
    counts = ak.run_lengths(data.evtid)
    return ak.unflatten(data, counts)


def group_by_time(obj, window=10):
    runs = np.array(np.cumsum(ak.run_lengths(obj.evtid)))
    counts = ak.run_lengths(obj.evtid)

    time_diffs = np.diff(obj.time)
    index_diffs = np.diff(obj.evtid)

    change_points = np.array(np.where((time_diffs > window * 1000) & (index_diffs == 0)))[0]
    total_change = np.sort(np.concatenate(([0], change_points, runs), axis=0))

    counts = ak.Array(np.diff(total_change))
    return ak.unflatten(obj, counts)


def sum_energy(grouped):
    sum_energy = ak.sum(grouped.edep, axis=-1)
    t0 = ak.fill_none(ak.firsts(grouped.time, axis=-1), np.nan)
    index = ak.fill_none(ak.firsts(grouped.evtid, axis=-1), np.nan)

    return ak.zip({"sum_energy": sum_energy, "t0": t0, "evtid": index})


def smear_energy(data, reso=2, energy_name="sum_energy"):
    return ak.with_field(
        data, np.random.Generator.normal(data[energy_name], reso), "energy_smeared"
    )
