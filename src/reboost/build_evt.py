from __future__ import annotations

import logging

import awkward as ak
import numpy as np
from dbetto import AttrsDict
from lgdo import Array, Table, VectorOfVectors, lh5

from . import core, math, shape, utils
from .shape import group

log = logging.getLogger(__name__)


def build_evt(
    tcm: VectorOfVectors,
    hitfile: str,
    outfile: str | None,
    channel_groups: AttrsDict,
    pars: AttrsDict,
    run_part: AttrsDict,
) -> Table | None:
    """Build events out of a TCM.

    Parameters
    ----------
    tcm
        the time coincidence map.
    hitfile
        file with the hits.
    outfile
        the path to the output-file, if `None` with return
        the events in memory.
    channel_groups
        a dictionary of groups of channels. For example:

        .. code-block:: python

            {"det1": "on", "det2": "off", "det3": "ac"}

    pars
        A dictionary of parameters. The first key should
        be the run ID, followed by different sets of parameters
        arranged in groups. Run numbers should be given in the
        format `"p00-r001"`, etc.

        For example:

        .. code-block:: python

            {"p03-r000": {"reso": {"det1": [1, 2], "det2": [0, 1]}}}

    run_part
        The run partitioning file giving the number of events
        for each run. This should be organized as a dictionary
        with the following format:

        .. code-block:: python

            {"p03-r000": 1000, "p03-r001": 2000}

    Returns
    -------
    the event file in memory as a table if no output file is specified.
    """
    tcm_tables = utils.get_table_names(tcm)
    tcm_ak = tcm.view_as("ak")

    # loop over the runs
    cum_sum = 0
    tab = None

    for idx, (run_full, n_event) in enumerate(run_part.items()):
        period, run = run_full.split("-")
        pars_tmp = pars[run_full]

        # create an output table
        out_tab = Table(size=n_event)

        tcm_tmp = tcm_ak[cum_sum : cum_sum + n_event]

        # usabilities

        is_off = shape.group.get_isin_group(
            tcm_tmp.table_key, channel_groups, tcm_tables, group="off"
        )

        # filter out off channels
        channels = tcm_tmp.table_key[~is_off]
        rows = tcm_tmp.row_in_table[~is_off]
        out_tab.add_field("channel", VectorOfVectors(channels))
        out_tab.add_field("row_in_table", VectorOfVectors(rows))

        out_tab.add_field("period", Array(np.ones(len(channels)) * int(period[1:])))
        out_tab.add_field("run", Array(np.ones(len(channels)) * int(run[1:])))

        # now check for channels in ac
        is_good = group.get_isin_group(channels, channel_groups, tcm_tables, group="on")

        # get energy
        energy_true = core.read_data_at_channel_as_ak(
            channels, rows, hitfile, "energy", "hit", tcm_tables
        )

        energy = math.stats.apply_energy_resolution(
            energy_true,
            channels,
            tcm_tables,
            pars_tmp.reso,
            lambda energy, sig0, sig1: np.sqrt(energy * sig1**2 + sig0**2),
        )

        out_tab.add_field("is_good", VectorOfVectors(is_good[energy > 25]))

        out_tab.add_field("energy", VectorOfVectors(energy[energy > 25]))
        out_tab.add_field("multiplicity", Array(ak.sum(energy > 25, axis=-1).to_numpy()))

        # write table
        wo_mode = "of" if idx == 0 else "append"

        # add attrs
        out_tab.attrs["tables"] = tcm.attrs["tables"]

        if outfile is not None:
            lh5.write(out_tab, "evt", outfile, wo_mode=wo_mode)
        else:
            tab = (
                ak.concatenate((tab, out_tab.view_as("ak")))
                if tab is not None
                else out_tab.view_as("ak")
            )

    return Table(tab)
