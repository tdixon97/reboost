"""A program for combining the hits from various detectors, to build events.

Is able to parse a config file with the following format config file:

.. code-block:: yaml

    channels:
        geds_on:
        - det001
        - det002
        geds_ac:
        - det003

    outputs:
    - energy
    - multiplicity

    operations:
     energy_id:
        channels: geds_on
        aggregation_mode: gather
        query: "hit.energy > 25"
        expression: tcm.channel_id

     energy:
        aggregation_mode: keep_at_ch:evt.energy_id
        expression: "hit.energy > 25"
        channels: geds_on

     multiplicity:
        channels: geds_on
        aggregation_mode: sum
        expression: "hit.energy > 25"
        initial: 0


Must contain:
- "channels": dictionary of channel groupings
- "outputs": fields for the output file
- "operations": operations to perform see :func:`pygama.evt.build_evt.evaluate_expression` for more details.
"""

from __future__ import annotations

import logging

import awkward as ak
import numpy as np
from lgdo import Table
from lgdo.lh5 import LH5Iterator, write
from pygama.evt.build_evt import evaluate_expression
from pygama.evt.utils import TCMData

from . import utils

log = logging.getLogger(__name__)


def build_evt(
    hit_file: str, tcm_file: str, evt_file: str | None, config: dict, buffer: int = int(5e6)
) -> ak.Array | None:
    """Generates the event tier from the hit and tcm.

    Parameters
    ----------
    hit_file
        path to the hit tier file
    tcm_file
        path to the tcm tier file
    evt_file
        path to the evt tier (output) file, if `None` the :class:`Table` is returned in memory
    config
        dictionary of the configuration.

    buffer
        number of events to process simultaneously

    Returns
    -------
    ak.Array of the evt tier data (if the data is not saved to disk)
    """
    msg = "... beginning the evt tier processing"
    log.info(msg)

    # create the objects needed for evaluate expression

    file_info = {
        "hit": (hit_file, "hit", "det{:03}"),
        "evt": (evt_file, "evt"),
    }

    # iterate through the TCM

    out_ak = ak.Array([])
    mode = "of"

    # get channel groupings
    channels = {}
    for group, info in config["channels"].items():
        if isinstance(info, str):
            channels[group] = [info]

        elif isinstance(info, list):
            channels[group] = info

    for tcm_lh5, _, n_rows_read in LH5Iterator(tcm_file, "tcm", buffer_len=buffer):
        tcm_lh5_sel = tcm_lh5
        tcm_ak = tcm_lh5_sel.view_as("ak")[:n_rows_read]

        tcm = TCMData(
            id=np.array(ak.flatten(tcm_ak.array_id)),
            idx=np.array(ak.flatten(tcm_ak.array_idx)),
            cumulative_length=np.array(np.cumsum(ak.num(tcm_ak.array_id, axis=-1))),
        )

        n_rows = len(tcm.cumulative_length)
        out_tab = Table(size=n_rows)

        for name, info in config["operations"].items():
            msg = f"computing field {name}"
            log.debug(msg)

            defaultv = info.get("initial", np.nan)
            if isinstance(defaultv, str) and (defaultv in ["np.nan", "np.inf", "-np.inf"]):
                defaultv = eval(defaultv)

            channels_use = utils.get_channels_from_groups(info.get("channels", []), channels)
            channels_exclude = utils.get_channels_from_groups(
                info.get("exclude_channels", []), channels
            )

            if "aggregation_mode" not in info:
                field = out_tab.eval(
                    info["expression"].replace("evt.", ""), info.get("parameters", {})
                )
            else:
                field = evaluate_expression(
                    file_info,
                    tcm,
                    channels_use,
                    table=out_tab,
                    mode=info["aggregation_mode"],
                    expr=info["expression"],
                    query=info.get("query", None),
                    sorter=info.get("sort", None),
                    channels_skip=channels_exclude,
                    default_value=defaultv,
                    n_rows=n_rows,
                )

            msg = f"field {field}"
            log.debug(msg)
            out_tab.add_field(name, field)

        # remove fields if necessary
        existing_cols = list(out_tab.keys())
        for col in existing_cols:
            if col not in config["outputs"]:
                out_tab.remove_column(col, delete=True)

        # write
        if evt_file is not None:
            write(out_tab, "evt", evt_file, wo_mode=mode)
            mode = "append"
        else:
            out_ak = ak.concatenate((out_ak, out_tab.view_as("ak")))

    if evt_file is None:
        return out_ak
    return None
