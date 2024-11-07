from __future__ import annotations

import logging

import numpy as np
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
    table: Table, info: dict, pars: dict
) -> Array | ArrayOfEqualSizedArrays | VectorOfVectors:
    """Evaluate an expression returning an LGDO object.

    Parameters
    ----------
    table
        hit table, with columns possibly used in the operations.
    info
        `dict` containing the information on the expression. Must contain `mode` and `expressions` keys
        For example:

        .. code-block:: json

            {
              "mode": "eval",
              "expression":"ak.sum(hit.edep,axis=-1)"
            }

        variables preceded by `hit` will be taken from the supplied table. Mode can be either `eval`,
        in which case a simple expression is based (only involving numpy, awkward or inbuilt python functions),
        or `function` in which case an arbitrary function is passed (for example defined in processors).

    pars
        dict of parameters, can contain any fields passed to the expression prefixed by `pars.`.


    Returns
    -------
    a new column for the hit table either :class:`Array`, :class:`ArrayOfEqualSizedArrays` or :class:`VectorOfVectors`.
    """

    pars_tuple = utils.dict2tuple(pars)
    local_dict = {"pars": pars_tuple}

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
    macro: str | None = None,
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
                        "fccd": 0.1
                    }
                }

        buffer
            length of buffer
        gdml
            path to the input gdml file.
        macro
            path to the macro file.

    Note
    ----
    The operations can depend on the outputs of previous steps, so operations order is important.
    """
    if gdml is not None:
        pass

    if macro is not None:
        pass

    for ch_idx, d in enumerate(proc_config["channels"]):
        msg = f"...running hit tier for {d}"
        log.info(msg)
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

                col = eval_expression(grouped, info, pars)
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
