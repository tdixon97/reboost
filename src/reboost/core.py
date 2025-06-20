from __future__ import annotations

import logging
import time
from typing import Any

import awkward as ak
import numpy as np
from dbetto import AttrsDict
from lgdo import lh5
from lgdo.types import LGDO, Table

from . import utils
from .profile import ProfileDict

log = logging.getLogger(__name__)


def read_data_at_channel_as_ak(
    channels: ak.Array, rows: ak.Array, file: str, field: str, group: str, tab_map: dict[int, str]
) -> ak.Array:
    r"""Read the data from a particular field to an awkward array. This replaces the TCM like object defined by the channels and rows with the corresponding data field.

    Parameters
    ----------
    channels
        Array of the channel indices (uids).
    rows
        Array of the rows in the files to gather data from.
    file
        File to read the data from.
    field
        the field to read.
    group
        the group to read data from (eg. `hit` or `stp`.)
    tab_map
        mapping between indices and table names. Of the form:

        .. code:: python

            {NAME: UID}

        For example:

        .. code:: python

            {"det001": 1, "det002": 2}

    Returns
    -------
    an array with the data, of the same same as the channels and rows.
    """
    # initialise the output
    data_flat = None
    tcm_rows_full = None

    # save the unflattening
    reorder = ak.num(rows)

    for tab_name, key in tab_map.items():
        # get the rows to read

        idx = ak.flatten(rows[channels == key]).to_numpy()
        arg_idx = np.argsort(idx)

        # get the rows in the flattened data we want to append to
        tcm_rows = np.where(ak.flatten(channels == key))[0]

        # read the data with sorted idx
        data_ch = lh5.read(f"{group}/{tab_name}/{field}", file, idx=idx[arg_idx]).view_as("ak")

        # sort back to order for tcm
        data_ch = data_ch[np.argsort(arg_idx)]

        # append to output
        data_flat = ak.concatenate((data_flat, data_ch)) if data_flat is not None else data_ch
        tcm_rows_full = (
            np.concatenate((tcm_rows_full, tcm_rows)) if tcm_rows_full is not None else tcm_rows
        )

    if len(data_flat) != len(tcm_rows_full):
        msg = "every index in the tcm should have been read"
        raise ValueError(msg)

    # sort the final data
    data_flat = data_flat[np.argsort(tcm_rows_full)]

    return ak.unflatten(data_flat, reorder)


def evaluate_output_column(
    hit_table: Table,
    expression: str,
    local_dict: dict,
    *,
    table_name: str = "HITS",
    time_dict: ProfileDict | None = None,
    name: str = " ",
) -> LGDO:
    """Evaluate an expression returning an LGDO.

    Uses :func:`lgdo.Table.eval()` to compute a new column for the
    hit table. The expression can depend on any field in the Table
    (prefixed with table_name.) or objects contained in the local dict.
    In addition, the expression can use packages which are then imported.

    Parameters
    ----------
    hit_table
        the table containing the hit fields.
    expression
        the expression to evaluate.
    local_dict
        local dictionary to pass to :func:`lgdo.Table.eval()`.
    table_name
        keyword used to refer to the fields in the table.
    time_dict
        time profiling data structure.
    name
        name to use in `time_dict`.

    Returns
    -------
    an LGDO with the new field.
    """
    if time_dict is not None:
        time_start = time.time()

    if local_dict is None:
        local_dict = {}

    expr = expression.replace(f"{table_name}.", "")

    # get func call and modules to import

    func_call, globals_dict = utils.get_function_string(expr)

    msg = f"evaluating table with command {expr} and local_dict {local_dict.keys()}"
    log.debug(msg)

    # remove np and ak
    globals_dict.pop("np", None)
    globals_dict.pop("ak", None)

    if globals_dict == {}:
        globals_dict = None

    if globals_dict is not None and "pyg4ometry" in globals_dict:
        with utils.filter_logging(logging.CRITICAL):
            res = hit_table.eval(func_call, local_dict, modules=globals_dict)
    else:
        res = hit_table.eval(func_call, local_dict, modules=globals_dict)

    # how long did it take
    if time_dict is not None:
        time_dict.update_field(name=f"expressions/{name}", time_start=time_start)

    return res


def evaluate_object(
    expression: str,
    local_dict: dict,
) -> Any:
    """Evaluate an expression returning any object.

    The expression should be a function call. It can depend on any objects contained in the local dict.
    In addition, the expression can use packages which are then imported.

    Parameters
    ----------
    expression
        the expression to evaluate.
    local_dict
        local dictionary to pass to `eval()`.

    Returns
    -------
    the evaluated object.
    """
    msg = f"Evaluating object with expression {expression} and {local_dict}"
    log.debug(msg)

    func_call, globals_dict = utils.get_function_string(expression)

    if "pyg4ometry" in globals_dict:
        with utils.filter_logging(logging.CRITICAL):
            return eval(func_call, local_dict, globals_dict)
    else:
        return eval(func_call, local_dict, globals_dict)


def get_global_objects(
    expressions: dict[str, str], *, local_dict: dict, time_dict: dict | None = None
) -> AttrsDict:
    """Extract global objects used in the processing.

    Parameters
    ----------
    expressions
        a dictionary containing the expressions to evaluate for each object.
    local_dict
        other objects used in the evaluation of the expressions, passed to
        `eval()` as the locals keyword.
    time_dict
        time profiling data structure.

    Returns
    -------
    dictionary of objects with the same keys as the expressions.
    """
    if time_dict is not None:
        time_start = time.time()

    msg = f"Getting global objects with {expressions.keys()} and {local_dict}"
    log.debug(msg)
    res = {}

    for obj_name, expression in expressions.items():
        res[obj_name] = evaluate_object(
            expression, local_dict=local_dict | {"OBJECTS": AttrsDict(res)}
        )

    if time_dict is not None:
        time_dict.update_field(name="global_objects", time_start=time_start)

    return AttrsDict(res)


def get_detector_mapping(detector_mapping: dict, global_objects: AttrsDict) -> dict:
    """Get all the detector mapping using :func:`get_one_detector_mapping`.

    Parameters
    ----------
    detector_mapping
        dictionary of detector mapping
    global_objects
        dictionary of global objects to use in evaluating the mapping.
    """
    return utils.merge_dicts(
        [
            get_one_detector_mapping(
                mapping["output"],
                input_detector_name=mapping.get("input", None),
                objects=global_objects,
            )
            for mapping in detector_mapping
        ]
    )


def get_one_detector_mapping(
    output_detector_expression: str | list,
    objects: AttrsDict | None = None,
    input_detector_name: str | None = None,
) -> dict:
    """Extract the output detectors and the list of input to outputs by parsing the expressions.

    The output_detector_expression can be a name or a string evaluating to a list of names.
    This expression can depend on any objects in the objects dictionary, referred to by the keyword
    "OBJECTS".

    The function produces a dictionary mapping input detectors to output detectors with the following
    format:

    .. code-block:: python

        {
            "input1": ["output1", "output2"],
            "input2": ["ouput3", ...],
        }

    If only output_detector_expression is supplied the mapping is one-to-one (i.e. every
    input detector maps to the same output detector). If instead a name for the input_detector_name
    is also supplied this will be the only key with all output detectors being mapped to this.

    Parameters
    ----------
    output_detector_expression
        An output detector name or a string evaluating to a list of output tables.
    objects
        dictionary of objects that can be referenced in the expression.
    input_detector_name
        Optional input detector name for all the outputs.


    Returns
    -------
    a dictionary with the input detectors as key and a list of output detectors for each.

    Examples
    --------
    For a direct one-to-one mapping:

    >>> get_detectors_mapping("[str(i) for i in range(2)]")
    {'0':['0'],'1':['1'],'2':['2']}

    With an input detector name:

    >>> get_detectors_mapping("[str(i) for i in range(2)])",input_detector_name = "dets")
    {'dets':['0','1','2']}

    With  objects:

    >>> objs = AttrsDict({"format": "ch"})
    >>> get_detectors_mapping("[f'{OBJECTS.format}{i}' for i in range(2)])",
                                input_detector_name = "dets",objects=objs)
    {'dets': ['ch0', 'ch1', 'ch2']}
    """
    out_names = []
    if isinstance(output_detector_expression, str):
        out_list = [output_detector_expression]
    else:
        out_list = list(output_detector_expression)

    for expression_tmp in out_list:
        func, globs = utils.get_function_string(expression_tmp)

        # if no package was imported its just a name
        try:
            objs = evaluate_object(expression_tmp, local_dict={"OBJECTS": objects})
            out_names.extend(objs)
        except Exception:
            out_names.append(expression_tmp)

    # simple one to one mapping
    if input_detector_name is None:
        return {name: [name] for name in out_names}
    return {input_detector_name: out_names}


def get_detector_objects(
    output_detectors: list,
    expressions: dict,
    args: AttrsDict,
    global_objects: AttrsDict,
    time_dict: ProfileDict | None = None,
) -> AttrsDict:
    """Get the detector objects for each detector.

    This computes a set of objects per output detector. These should be the
    expressions (defined in the `expressions` input). They can depend
    on the keywords:

    - `ARGS` : in which case values of from the args parameter AttrsDict can be references,
    - `DETECTOR`: referring to the detector name (key of the detector mapping)
    - `OBJECTS` : The global objects.

    For example expressions like:

    .. code-block:: python

        compute_object(arg=ARGS.first_arg, detector=DETECTOR, obj=OBJECTS.meta)

    are supported.

    Parameters
    ----------
    output_detectors
        list of output detectors,
    expressions
        dictionary of expressions to evaluate.
    args
        any arguments the expression can depend on, is passed as `locals` to `eval()`.
    global_objects
        a dictionary of objects the expression can depend on.
    time_dict
        time profiling data structure.

    Returns
    -------
    An AttrsDict of the objects for each detector.
    """
    if time_dict is not None:
        time_start = time.time()

    det_objects_dict = {}
    for output_detector in output_detectors:
        obj_dict = {}
        for obj_name, obj_expression in expressions.items():
            obj_dict[obj_name] = evaluate_object(
                obj_expression,
                local_dict={
                    "ARGS": args,
                    "DETECTOR": output_detector,
                    "OBJECTS": global_objects,
                    "DETECTOR_OBJECTS": AttrsDict(obj_dict),
                },
            )

        det_objects_dict[output_detector] = AttrsDict(obj_dict)
    res = AttrsDict(det_objects_dict)

    if time_dict is not None:
        time_dict.update_field(name="detector_objects", time_start=time_start)

    return res


def evaluate_hit_table_layout(
    steps: ak.Array | Table, expression: str, time_dict: dict | None = None
) -> Table:
    """Evaluate the hit_table_layout expression, producing the hit table.

    This expression should be a function call which performs a restructuring of the steps,
    i.e. it sets the number of rows. The steps array should be referred to
    by "STEPS" in the expression.

    Parameters
    ----------
    steps
        awkward array or Table of the steps.
    expression
        the expression to evaluate to produce the hit table.
    time_dict
        time profiling data structure.

    Returns
    -------
    :class:`lgdo.Table` of the hits.
    """
    if time_dict is not None:
        time_start = time.time()

    group_func, globs = utils.get_function_string(
        expression,
    )
    locs = {"STEPS": steps}

    msg = f"running step grouping with {group_func} and globals {globs.keys()} and locals {locs.keys()}"
    log.debug(msg)

    res = eval(group_func, globs, locs)

    if time_dict is not None:
        time_dict.update_field(name="hit_layout", time_start=time_start)

    return res


def add_field_with_nesting(tab: Table, col: str, field: LGDO) -> Table:
    """Add a field handling the nesting."""
    subfields = col.strip("/").split("___")
    tab_next = tab

    for level in subfields:
        # if we are at the end, just add the field
        if level == subfields[-1]:
            tab_next.add_field(level, field)
            break

        if not level:
            msg = f"invalid field name '{field}'"
            raise RuntimeError(msg)

        # otherwise, increase nesting
        if level not in tab:
            tab_next.add_field(level, Table(size=len(tab)))
            tab_next = tab[level]
        else:
            tab_next = tab[level]

    return tab


def _get_table_keys(tab: Table):
    """Get keys in a table."""
    existing_cols = list(tab.keys())
    output_cols = []
    for col in existing_cols:
        if isinstance(tab[col], Table):
            output_cols.extend(
                [f"{col}___{col_second}" for col_second in _get_table_keys(tab[col])]
            )
        else:
            output_cols.append(col)

    return output_cols


def _remove_col(field: str, tab: Table):
    """Remove column accounting for nesting."""
    if "___" in field:
        base_name, sub_field = field.split("___", 1)[0], field.split("___", 1)[1]
        _remove_col(sub_field, tab[base_name])
    else:
        tab.remove_column(field, delete=True)


def remove_columns(tab: Table, outputs: list) -> Table:
    """Remove columns from the table not found in the outputs.

    Parameters
    ----------
    tab
        the table to remove columns from.
    outputs
        a list of output fields.

    Returns
    -------
    the table with columns removed.
    """
    cols = _get_table_keys(tab)
    for col_unrename in cols:
        if col_unrename not in outputs:
            _remove_col(col_unrename, tab)
    return tab


def merge(hit_table: Table, output_table: ak.Array | None):
    """Merge the table with the array."""
    return (
        hit_table.view_as("ak")
        if output_table is None
        else ak.concatenate((output_table, hit_table.view_as("ak")))
    )
