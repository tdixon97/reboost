from __future__ import annotations

import logging
import re

import awkward as ak
from lgdo import Table, lh5

from reboost import shape

log = logging.getLogger(__name__)


def build_tcm(
    hit_file: str,
    out_file: str,
    channels: list[str],
    time_name: str = "t0",
    idx_name: str = "global_evtid",
    time_window_in_us: float = 10,
) -> None:
    """Build the (Time Coincidence Map) TCM from the hit tier.

    Parameters
    ----------
    hit_file
        path to hit tier file.
    out_file
        output path for tcm.
    channels
        list of channel names to include.
    time_name
        name of the hit tier field used for time grouping.
    idx_name
        name of the hit tier field used for index grouping.
    time_window_in_us
        time window used to define the grouping.
    """
    hash_func = r"\d+"

    msg = "start building time-coincidence map"
    log.info(msg)

    chan_ids = [re.search(hash_func, channel).group() for channel in channels]

    hit_data = []
    for channel in channels:
        hit_data.append(
            lh5.read(f"{channel}/hit", hit_file, field_mask=[idx_name, time_name]).view_as("ak")
        )
    tcm = get_tcm_from_ak(
        hit_data, chan_ids, window=time_window_in_us, time_name=time_name, idx_name=idx_name
    )

    if tcm is not None:
        lh5.write(tcm, "tcm", out_file, wo_mode="of")


def get_tcm_from_ak(
    hit_data: list[ak.Array],
    channels: list[int],
    *,
    window: float = 10,
    time_name: str = "t0",
    idx_name: str = "global_evtid",
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

        obj_tmp = ak.with_field(obj_tmp, hit_idx, "array_idx")

        obj_tmp["array_id"] = int(ch_idx)
        sort_objs.append(obj_tmp)

    obj_tot = ak.concatenate(sort_objs)

    return shape.group.group_by_time(
        obj_tot,
        time_name=time_name,
        evtid_name=idx_name,
        window=window,
        fields=["array_id", "array_idx"],
    )
