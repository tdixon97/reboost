from __future__ import annotations

import logging
from typing import Any

import awkward as ak
from dbetto import AttrsDict
from lgdo.types import LGDO, Table

from . import utils

log = logging.getLogger(__name__)


def evaluate_expression(
    hit_table: Table,
    expression: str,
    local_dict: dict,
    *,
    table_name: str = "HITS",
) -> LGDO:
    """Evaluate an expression returning an LGDO.

    Uses :func:`LGDO.Table.eval()` to compute a new column for the
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
        local dictionary to pass to :func:`LGDO.Table.eval()`.
    table_name
        keyword used to refer to the fields in the table.

    Returns
    -------
        an LGDO with the new field.
    """

    if local_dict is None:
        local_dict = {}

    expr = expression.replace(f"{table_name}.", "")

    # get func call and modules to import
    func_call, globals_dict = utils.get_function_string(expr)

    msg = f"evaluating table with command {expr} and local_dict {local_dict.keys()}"
    log.debug(msg)

    return hit_table.eval(func_call, local_dict, globals=globals_dict)


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
        local dictionary to pass to :func:`LGDO.Table.eval()`.

    Returns
    -------
        the evaluated object.
    """
    func_call, globals_dict = utils.get_function_string(expression)
    return eval(func_call, local_dict, globals_dict)


def get_global_objects(expressions: dict[str, str], *, local_dict: dict) -> dict:
    """Extract global objects used in the processing.

    Parameters
    ----------
    expressions
        a dictionary containing the expressions to evaluate for each object.
    local_dict
        other objects used in the evaluation of the expressions, passed to
        `eval()` as the locals keyword.

    Returns
    -------
    dictionary of objects with the same keys as the expressions.
    """
    return AttrsDict(
        {
            obj_name: evaluate_object(expression, local_dict=local_dict)
            for obj_name, expression in expressions.items()
        }
    )


def get_detectors_mapping(
    output_detector_expression: str, input_detector_map_expression: str | None, objects: dict
) -> dict:
    """Extract the output detectors and the list of input to outputs."""
    raise NotImplementedError


def get_detector_objects(output_detectors, proc_group, args, global_objects) -> AttrsDict:
    """Get the detector objects"""
    det_objects_dict = {}
    for output_detector in output_detectors:
        det_objects_dict[output_detector] = AttrsDict(
            {
                obj_name: evaluate_object(
                    obj_expression,
                    local_dict={
                        "ARGS": args,
                        "DETECTOR": output_detector,
                        "OBJECTS": global_objects,
                    },
                )
                for obj_name, obj_expression in proc_group.get("detector_objects", {}).items()
            }
        )
    return AttrsDict(det_objects_dict)


def evaluate_hit_table_layout(steps: ak.Array | Table, expression: str) -> Table:
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

    Returns
    -------
    :class:`LGDO.Table` of the hits.
    """

    group_func, globs = utils.get_function_string(
        expression,
    )
    locs = {"STEPS": steps}

    msg = f"running step grouping with {group_func} and globals {globs.keys()} and locals {locs.keys()}"
    log.debug(msg)

    return eval(group_func, globs, locs)


def remove_columns(tab: Table, outputs: list) -> Table:
    """Remove columns from the table not found in the outputs

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

    existing_cols = list(tab.keys())
    for col in existing_cols:
        if col not in outputs:
            tab.remove_column(col, delete=True)

    return tab
