from __future__ import annotations

import logging

import awkward as ak
from lgdo import Table

from . import processors

log = logging.getLogger(__name__)


def build_tcm(
    hit_data: list[ak.Array],
    channels: list[int],
    *,
    window: float = 10,
    time_name: str = "t0",
    idx_name: str = "hit_global_evtid",
) -> Table:
    """Builds a time-coincidence map from a hit of hit data Tables.

    - build an ak.Array of the data merging channels with fields base on "time_name", and "idx_name" and adding a field `rawid` from the channel idx, also add the row (`hit_idx`)
    - sorts this array by "idx_name" then "time_name" fields
    - group by "idx_name" and "time_name" based on the window parameter

    Parameters
    ----------
    hit_data
        list of hit tier data for each channel
    channels
        list of channel indices
    window
        time window for selecting coincidences (in us)
    time_name
        name of the field for time information
    idx_name
        name of the decay index field

    Returns
    -------
    an LGDO.VectorOfVectors containing the time-coincidence map
    """
    # build ak_obj for sorting
    sort_objs = []
    for ch_idx, data_tmp in zip(channels, hit_data):
        obj_tmp = ak.copy(data_tmp)
        obj_tmp = obj_tmp[[time_name, idx_name]]
        hit_idx = ak.local_index(obj_tmp)

        obj_tmp = ak.with_field(obj_tmp, hit_idx, "hit_idx")

        obj_tmp["rawid"] = ch_idx

        sort_objs.append(obj_tmp)

    obj_tot = ak.concatenate(sort_objs)

    return processors.group_by_time(
        obj_tot,
        time_name=time_name,
        evtid_name=idx_name,
        window=window,
        fields=["rawid", "hit_idx"],
    )
