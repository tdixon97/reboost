from __future__ import annotations

import awkward as ak
import numpy as np


def group_by_event(data):
    counts = ak.run_lengths(data["evtid"])

    grouped = ak.unflatten(data, counts)
    sum_energy = ak.sum(grouped.edep, axis=-1)

    t0 = ak.fill_none(ak.firsts(grouped.time, axis=-1), np.nan)
    index = ak.fill_none(ak.firsts(grouped.evtid, axis=-1), np.nan)

    return ak.zip({"sum_energy": sum_energy, "t0": t0, "evtid": index})


def smear_energy(energies, reso_func):
    return np.random.Generator.normal(energies, reso_func(energies))
