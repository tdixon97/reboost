from __future__ import annotations

import logging

import legendhpges
import numpy as np
import pyg4ometry
from lgdo import Array, ArrayOfEqualSizedArrays, LH5Iterator, Table, VectorOfVectors, lh5

from . import utils

log = logging.getLogger(__name__)


def step_group(data: Table, group_config: dict) -> Table:
    """Performs a grouping of geant4 steps to build the `hit` table.

    Parameters
    ----------
    data
        `stp` table from remage.
    group_config
        dict with the configuration describing the step grouping.
        For example

        .. code-block:: json

            {
            "description": "group steps by time and evtid.",
            "expression": "reboost.hpge.processors.group_by_time(stp,window=10)"
            }

        this will then evaluate the function chosen by `expression` resulting in a new Table.

    """

    # group to create hit table

    group_func, globs = utils.get_function_string(
        group_config["expression"],
    )
    locs = {"stp": data}

    msg = f"running step grouping with {group_func} and globals {globs.keys()} and locals {locs.keys()}"
    log.debug(msg)
    return eval(group_func, globs, locs)


def eval_expression(
    table: Table,
    info: dict,
    pars: dict | None = None,
    hpge: legendhpges.HPGe | None = None,
    phy_vol: pyg4ometry.geant4.PhysicalVolume | None = None,
) -> Array | ArrayOfEqualSizedArrays | VectorOfVectors:
    """Evaluate an expression returning an LGDO object.

    Parameters
    ----------
    table
        hit table, with columns possibly used in the operations.
    info
        `dict` containing the information on the expression. Must contain `mode` and `expressions` keys. For example:

        .. code-block:: json

            {
              "mode": "eval",
              "expression":"ak.sum(hit.edep,axis=-1)"
            }

        Variables preceded by `hit` will be taken from the supplied table. Mode can be either `eval`,
        in which case a simple expression is based (only involving numpy, awkward or inbuilt python functions),
        or `function` in which case an arbitrary function is passed (for example defined in processors).

        There are several objects passed to the evaluation as 'locals' which can be references by the expression.
         - `pars`: dictionary of parameters (converted to namedtuple) (see `pars` argument),
         - `hpge`: the legendhpges object for this detector (see `hpge` argument),
         - `phy_vol`:  the physical volume of the detector (see `phy` argument).

    pars
        dict of parameters, can contain any fields passed to the expression prefixed by `pars.`.
    hpge
        `legendhpges` object with the information on the HPGe detector.
    phy_vol
        `pyg4ometry.geant4.PhysicalVolume` object from GDML,

    Returns
    -------
    a new column for the hit table either :class:`Array`, :class:`ArrayOfEqualSizedArrays` or :class:`VectorOfVectors`.

    Note
    ----
    In future the passing of local variables (pars,hpge,reg) to the evaluation should be make more generic.
    """
    local_dict = {}

    if pars is not None:
        pars_tuple = utils.dict2tuple(pars)
        local_dict = {"pars": pars_tuple}
    if phy_vol is not None:
        local_dict |= {"phy_vol": phy_vol}
    if hpge is not None:
        local_dict |= {"hpge": hpge}

    if info["mode"] == "eval":
        # replace hit.
        expr = info["expression"].replace("hit.", "")

        msg = f"evaluating table with command {expr} and local_dict {local_dict.keys()}"
        log.debug(msg)

        col = table.eval(expr, local_dict)

    elif info["mode"] == "function":
        proc_func, globs = utils.get_function_string(info["expression"])

        # add hit table to locals
        local_dict = {"hit": table} | local_dict

        msg = f"evaluating table with command {info['expression']} and local_dict {local_dict.keys()} and global dict {globs.keys()}"
        log.debug(msg)
        col = eval(proc_func, globs, local_dict)

    else:
        msg = "mode is not recognised."
        raise ValueError(msg)
    return col


def build_hit(
    file_out: str,
    file_in: str,
    out_field: str,
    in_field: str,
    proc_config: dict,
    pars: dict,
    buffer: int = 1000000,
    gdml: str | None = None,
    metadata_path: str | None = None,
) -> None:
    """
    Read incrementally the files compute something and then write output

    Parameters
    ----------
        file_out
            output file path
        file_in
            input_file_path
        out_field
            lh5 group name for output
        in_field
            lh5 group name for input
        proc_config
            the configuration file for the processing. Must contain the fields `channels`, `outputs`, `step_group` and operations`.
            For example:

            .. code-block:: json

               {
                    "channels": [
                        "det000",
                        "det001",
                        "det002",
                        "det003"
                    ],
                    "outputs": [
                        "t0",
                        "truth_energy_sum",
                        "smeared_energy_sum",
                        "evtid"
                    ],
                    "step_group": {
                        "description": "group steps by time and evtid.",
                        "expression": "reboost.hpge.processors.group_by_time(stp,window=10)"
                    },
                    "operations": {
                        "t0": {
                            "description": "first time in the hit.",
                            "mode": "eval",
                            "expression": "ak.fill_none(ak.firsts(hit.time,axis=-1),np.nan)"
                        },
                        "truth_energy_sum": {
                            "description": "truth summed energy in the hit.",
                            "mode": "eval",
                            "expression": "ak.sum(hit.edep,axis=-1)"
                        },
                        "smeared_energy_sum": {
                            "description": "summed energy after convolution with energy response.",
                            "mode": "function",
                            "expression": "reboost.hpge.processors.smear_energies(hit.truth_energy_sum,reso=pars.reso)"
                        }

                    }
                }

        pars
            a dictionary of parameters, must have a field per channel consisting of a `dict` of parameters. For example:

            .. code-block:: json

                {
                    "det000": {
                        "reso": 1,
                        "fccd": 0.1,
                        "phy_vol_name":"det_phy",
                        "meta_name": "icpc.json"
                    }
                }

            this should also contain the channel mappings needed by reboost. These are:
             - `phy_vol_name`: is the name of the physical volume,
             - `meta_name`    : is the name of the JSON file with the metadata.

            If these keys are not present both will be set to the remage output table name.

        buffer
            length of buffer
        gdml
            path to the input gdml file.
        macro
            path to the macro file.
        metadata_path
            path to the folder with the metadata (i.e. the `hardware.detectors.germanium.diodes` folder of `legend-metadata`)

    Note
    ----
     - The operations can depend on the outputs of previous steps, so operations order is important.
     - It would be better to have a cleaner way to supply metadata and detector maps.
    """
    # get the gdml file
    reg = pyg4ometry.gdml.Reader(gdml).getRegistry() if gdml is not None else None

    for ch_idx, d in enumerate(proc_config["channels"]):
        msg = f"...running hit tier for {d}"
        log.info(msg)

        # get HPGe and phy_vol object to pass to build_hit
        hpge = utils.get_hpge(metadata_path, pars=pars, detector=d)
        phy_vol = utils.get_phy_vol(reg, pars=pars, detector=d)

        delete_input = bool(ch_idx == 0)

        msg = f"...begin processing with {file_in} to {file_out}"
        log.info(msg)

        entries = LH5Iterator(file_in, f"{in_field}/{d}", buffer_len=buffer)._get_file_cumentries(0)

        # number of blocks is ceil of entries/buffer,
        # shift by 1 since idx starts at 0
        # this is maybe too high if buffer exactly divides idx
        max_idx = int(np.ceil(entries / buffer)) - 1
        buffer_rows = None

        for idx, (lh5_obj, _, _) in enumerate(
            LH5Iterator(file_in, f"{in_field}/{d}", buffer_len=buffer)
        ):
            msg = f"... processed {idx} files out of {max_idx}"
            log.debug(msg)

            # convert to awkward
            ak_obj = lh5_obj.view_as("ak")

            # handle the buffers
            obj, buffer_rows, mode = utils._merge_arrays(
                ak_obj, buffer_rows, idx=idx, max_idx=max_idx, delete_input=delete_input
            )

            # convert back to a table, should work
            data = Table(obj)

            # group steps into hits
            grouped = step_group(data, proc_config["step_group"])

            # processors
            for name, info in proc_config["operations"].items():
                msg = f"adding column {name}"
                log.debug(msg)

                col = eval_expression(grouped, info, pars=pars, phy=phy_vol, hpge=hpge)
                grouped.add_field(name, col)

            # remove unwanted columns
            log.debug("removing unwanted columns")

            existing_cols = list(grouped.keys())
            for col in existing_cols:
                if col not in proc_config["outputs"]:
                    grouped.remove_column(col, delete=True)

            # write lh5 file
            msg = f"...finished processing and save file with wo_mode {mode}"
            log.debug(msg)
            lh5.write(grouped, f"{out_field}/{d}", file_out, wo_mode=mode)
