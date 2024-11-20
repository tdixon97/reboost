from __future__ import annotations

import logging
import re

import awkward as ak
from lgdo import Table, lh5

from . import processors, utils

log = logging.getLogger(__name__)


def build_tcm(
    hit_file: str,
    out_file: str,
    channels: list[str],
    time_name: str = "t0",
    idx_name: str = "hit_global_evtid",
    time_window_in_us: float = 10,
    idx_buffer: int = int(1e7),
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
    idx_buffer
        number of evtid to read in simultaneously.


    Notes
    -----
    This function avoids excessive memory usage by iterating over the files select sets of evtid,
    as such it may be a bit wasteful of IO, since the same block may need to be read multiple times.
    Since only a few fields are read a high value of idx_buffer can be used.

    """

    hash_func = r"\d+"

    # get number of evtids
    n_evtid = utils.get_num_evtid_hit_tier(hit_file, channels, idx_name)

    # not iterate over evtid
    n_evtid_read = 0
    mode = "of"
    msg = "start building time-coincidence map"
    log.info(msg)

    chan_ids = [re.search(hash_func, channel).group() for channel in channels]

    while n_evtid_read < n_evtid:
        # object for the data

        msg = f"... iterating: selecting evtid {n_evtid_read} to {n_evtid_read+idx_buffer}"
        log.debug(msg)

        hit_data, n_evtid_read = utils.read_some_idx_as_ak(
            channels=channels,
            file=hit_file,
            n_idx_read=n_evtid_read,
            idx_buffer=idx_buffer,
            idx_name=idx_name,
            field_mask=[idx_name, time_name],
        )

        tcm = get_tcm_from_ak(
            hit_data, chan_ids, window=time_window_in_us, time_name=time_name, idx_name=idx_name
        )
        if tcm is not None:
            lh5.write(tcm, "tcm", out_file, wo_mode=mode)
            mode = "append"


def get_tcm_from_ak(
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

        obj_tmp["channel_id"] = int(ch_idx)
        sort_objs.append(obj_tmp)

    obj_tot = ak.concatenate(sort_objs)

    return processors.group_by_time(
        obj_tot,
        time_name=time_name,
        evtid_name=idx_name,
        window=window,
        fields=["channel_id", "hit_idx"],
    )
