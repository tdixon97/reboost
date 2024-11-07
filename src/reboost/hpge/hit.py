from __future__ import annotations

import logging

import numpy as np
from lgdo import Array, ArrayOfEqualSizedArrays, LH5Iterator, Table, VectorOfVectors, lh5

from . import utils

log = logging.getLogger(__name__)


def eval_expression(
    table: Table, info: dict, pars: dict
) -> Array | ArrayOfEqualSizedArrays | VectorOfVectors:
    """Evaluate an expression returning an LGDO object.

    Parameters
    ----------
    table
        hit table, with columns possibly used in the operations.
    info
        dict containing the information on the expression. Must contain `mode` and `expressions` keys
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
        a new column for the hit table either :class:`Array`,:class:`ArrayOfEqualSizedArrays` or :class:` VectorOfVectors`.
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


def read_write_incremental(
    file_out: str,
    name_out: str,
    proc_config: dict,
    pars: dict,
    field: str,
    file: str,
    buffer: int = 1000000,
    delete_input: bool = False,
) -> None:
    """
    Read incrementally the files compute something and then write output

    Parameters
    ----------
        file_out
            output file path
        name_out
            lh5 group name for output
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
                        "reso": 1.,
                        "fccd": 0.1
                    }
                }

        field
            lh5 field name to read.
        file
            file name to read
        buffer
            length of buffer
        delete_input
            flag to delete the input file.

    Note
    ----
    The operations can depend on the outputs of previous steps, so operations order is important.
    """

    msg = f"...begin processing with {file} to {file_out}"
    log.info(msg)

    entries = LH5Iterator(file, field, buffer_len=buffer)._get_file_cumentries(0)

    # number of blocks is ceil of entries/buffer,
    # shift by 1 since idx starts at 0
    # this is maybe too high if buffer exactly divides idx
    max_idx = int(np.ceil(entries / buffer)) - 1
    buffer_rows = None

    for idx, (lh5_obj, _, _) in enumerate(LH5Iterator(file, field, buffer_len=buffer)):
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

        # define glob and local dicts for the func cal
        group_func, globs = utils.get_function_string(
            proc_config["step_group"]["expression"],
        )
        locs = {"stp": data}

        msg = f"running step grouping with {group_func} and globals {globs.keys()} and locals {locs.keys()}"
        log.debug(msg)

        # group to create hit table
        grouped = eval(group_func, globs, locs)

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
        lh5.write(grouped, name_out, file_out, wo_mode=mode)


def build_hit(
    lh5_in_file: str,
    lh5_out_file: str,
    proc_config: dict,
    pars: dict,
    buffer_len: int = int(5e6),
    gdml: str | None = None,
    macro: str | None = None,
) -> None:
    """
    Build the hit tier from the raw Geant4 output

    Parameters
    ----------
    lh5_in_file
        input file containing the raw tier
    lh5_out_file
        output file
    config
        dictionary containing the configuration / parameters
        should contain one sub-dictonary per detector with a format like:

        .. code-block:: json

            "det001": {
                "reso": 1
            }

        This can contain any parameters needed in the processing chain.
    buffer_len
        number of rows to read at once
    gdml
        path to the gdml file of the geometry
    macro
        path to the Geant4 macro used to generate the raw tier
    """

    # get the geant4 gdml and macro

    if gdml is not None:
        pass

    if macro is not None:
        pass

    for idx, d in enumerate(proc_config["channels"]):
        msg = f"...running hit tier for {d}"
        log.info(msg)
        delete_input = bool(idx == 0)

        read_write_incremental(
            lh5_out_file,
            f"hit/{d}",
            proc_config,
            pars[d],
            f"hit/{d}",
            lh5_in_file,
            buffer_len,
            delete_input=delete_input,
        )
