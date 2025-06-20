from __future__ import annotations

import logging

import awkward as ak
import numpy as np
from dbetto import AttrsDict
from lgdo import Table, VectorOfVectors
from numpy.typing import ArrayLike

log = logging.getLogger(__name__)


def isin(channels: ak.Array, chan_list: list):
    """Check if each element of the awkward array channels is in the channel list."""
    num_channels = ak.num(channels, axis=-1)
    channels_flat = ak.flatten(channels)
    isin = np.isin(channels_flat, chan_list)

    # unflatten
    return ak.unflatten(isin, num_channels)


def get_isin_group(
    channels: ArrayLike, groups: AttrsDict, tcm_tables: dict, group: str = "off"
) -> ak.Array:
    """For each channel check if it is in the group.

    Parameters
    ----------
    channels
        Array of the channel indices.
    groups
        A mapping of the group for every channel name.
    tcm_tables
        the mapping of indices to table names
    group
        the group to select.

    Returns
    -------
    an awkward array of the same shape of channels of booleans.
    """
    usability = {uid: groups[name] for name, uid in tcm_tables.items()}
    group_idx = [key for key, item in usability.items() if item == group]

    return isin(channels, group_idx)


def _sort_data(obj: ak.Array, *, time_name: str = "time", evtid_name: str = "evtid") -> ak.Array:
    """Sort the data by evtid then time.

    Parameters
    ----------
    obj
        array of records containing fields `time` and `evtid`.
    time_name
        name of the time field in `obj`.
    evtid_name
        name of the evtid field in `obj`.

    Returns
    -------
    sorted awkward array
    """
    obj = obj[ak.argsort(obj[evtid_name])]
    obj_unflat = ak.unflatten(obj, ak.run_lengths(obj[evtid_name]))

    indices = ak.argsort(obj_unflat[time_name], axis=-1)
    sorted_obj = obj_unflat[indices]

    return ak.flatten(sorted_obj)


def group_by_evtid(data: Table | ak.Array, *, evtid_name: str = "evtid") -> Table:
    """Simple grouping by evtid.

    Takes the input `stp` :class:`lgdo.Table` from remage and defines groupings of steps (i.e the
    `cumulative_length` for a vector of vectors). This then defines the output table (also :class:`lgdo.Table`),
    on which processors can add fields.

    Parameters
    ----------
    data
        LGDO Table which must contain the `evtid` field.
    evtid_name
        the name of the index field in the input table.

    Returns
    -------
    LGDO table of :class:`VectorOfVector` for each field.

    Note
    ----
    The input table must be sorted (by `evtid`).
    """
    # convert to awkward
    obj_ak = data.view_as("ak") if isinstance(data, Table) else data

    # extract cumulative lengths
    counts = ak.run_lengths(obj_ak[evtid_name])
    cumulative_length = np.cumsum(counts)

    # convert to numpy
    if isinstance(cumulative_length, ak.Array):
        cumulative_length = cumulative_length.to_numpy()

    # build output table
    out_tbl = Table(size=len(cumulative_length))

    for f in obj_ak.fields:
        out_tbl.add_field(
            f,
            VectorOfVectors(
                cumulative_length=cumulative_length, flattened_data=obj_ak[f].to_numpy()
            ),
        )
    return out_tbl


def group_by_time(
    data: Table | ak.Array,
    window: float = 10,
    time_name: str = "time",
    evtid_name: str = "evtid",
    fields: list | None = None,
) -> Table:
    """Grouping of steps by `evtid` and `time`.

    Takes the input `stp` :class:`lgdo.Table` from remage and defines groupings of steps (i.e the
    `cumulative_length` for a vector of vectors). This then defines the output table (also :class:`lgdo.Table`),
    on which processors can add fields.

    The windowing is based on defining a new group when the `evtid` changes or when the time increases by `> window`,
    which is in units of us.

    Parameters
    ----------
    data
        :class:`lgdo.Table` or `ak.Array` which must contain the time_name and evtid_name fields
    window
        time window in us used to search for coincident hits
    time_name
        name of the timing field
    evtid_name
        name of the evtid field
    fields
        names of fields to include in the output table, if None includes all

    Returns
    -------
    LGDO table of :class:`VectorOfVector` for each field.

    Note
    ----
    The input table must be sorted (first by `evtid` then `time`).
    """
    obj = data.view_as("ak") if isinstance(data, Table) else data
    obj = _sort_data(obj, time_name=time_name, evtid_name=evtid_name)

    # get difference
    time_diffs = np.diff(obj[time_name])
    index_diffs = np.diff(obj[evtid_name])

    # index of the last element in each run
    time_change = (time_diffs > window * 1000) & (index_diffs == 0)
    index_change = index_diffs > 0

    # cumulative length is just the index of changes plus 1
    cumulative_length = np.array(np.where(time_change | index_change))[0] + 1

    # add the las grouping
    cumulative_length = np.append(cumulative_length, len(obj[time_name]))

    # convert to numpy
    if isinstance(cumulative_length, ak.Array):
        cumulative_length = cumulative_length.to_numpy()

    # build output table
    out_tbl = Table(size=len(cumulative_length))

    fields = obj.fields if fields is None else fields
    for f in fields:
        out_tbl.add_field(
            f,
            VectorOfVectors(cumulative_length=cumulative_length, flattened_data=obj[f].to_numpy()),
        )

    return out_tbl
