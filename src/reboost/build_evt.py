from __future__ import annotations

import logging

import awkward as ak
import numpy as np
from dbetto import AttrsDict, TextDB
from lgdo import Array, Table, VectorOfVectors, lh5

from . import core, math, shape, utils

log = logging.getLogger(__name__)


def build_evt(
    tcm: VectorOfVectors,
    hitfile: str,
    outfile: str | None,
    meta: TextDB,
    pars: AttrsDict,
    run_part: AttrsDict,
) -> ak.Array | None:
    """Build events out of a TCM.

    Parameters
    ----------
    tcm
        the time coincidence map.
    hitfile
        file with the hits.
    outfile
        the path to the output-file.
    meta
        the metadata database
    pars
        extra parameters
    run_part
        the run partitioning file.
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

        # get the channel map
        timestamp = meta.datasets.runinfo[period][run].phy.start_key
        chmap = meta.channelmap(timestamp)

        # usabilities

        is_off = shape.group.get_isin_group(tcm_tmp.table_key, chmap, tcm_tables, group="off")

        # filter out off channels
        channels = tcm_tmp.table_key[~is_off]
        rows = tcm_tmp.row_in_table[~is_off]

        out_tab.add_field("channel", VectorOfVectors(channels))
        out_tab.add_field("row_in_table", VectorOfVectors(rows))

        # now check for channels in ac
        is_good = shape.group.get_isin_group(channels, chmap, tcm_tables, group="on")

        # get energy
        energy_true = core.read_data_at_channel(channels, rows, hitfile, "energy", tcm_tables)

        energy = math.stats.apply_energy_resolution(
            energy_true,
            channels,
            tcm_tables,
            pars_tmp.reso,
            lambda energy, sig0, sig1: np.sqrt(energy * sig1**2 + sig0**2),
        )

        out_tab.add_field("is_good", VectorOfVectors(is_good[energy > 25]))

        out_tab.add_field("energy", VectorOfVectors(energy[energy > 25]))
        out_tab.add_field("multiplicity", Array(ak.sum(energy > 25, axis=-1)))

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

    return tab
