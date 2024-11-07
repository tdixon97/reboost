from __future__ import annotations

import copy
import importlib
import logging
import re
from collections import namedtuple

import awkward as ak

log = logging.getLogger(__name__)


def get_detector_origin(name):
    raise NotImplementedError


def dict2tuple(dictionary: dict) -> namedtuple:
    return namedtuple("parameters", dictionary.keys())(**dictionary)


def get_function_string(expr: str) -> tuple[str, dict]:
    """Get a function call to evaluate."""

    args_str = re.search(r"\((.*)\)$", expr.strip()).group(1)

    # get module and function names
    func_call = expr.strip().split("(")[0]
    subpackage, func = func_call.rsplit(".", 1)
    package = subpackage.split(".")[0]
    importlib.import_module(subpackage, package=__package__)

    # declare imported package as globals (see eval() call later)
    globs = {
        package: importlib.import_module(package),
    }

    call_str = f"{func_call}({args_str})"

    return call_str, globs


def _merge_arrays(
    ak_obj: ak.Array, buffer_rows: ak.Array, idx: int, max_idx: int, delete_input: bool = False
) -> tuple[ak.Array, ak.Array, str]:
    """Merge awkward arrays with a buffer and simultaneously remove the last rows.

    This function is used since when iterating through the rows of an Array it will
    sometimes happen to split some events. This concatenates rows in the buffer onto the start of the data.
    This also removes the last rows of each chunk and saves them into a buffer.

    Parameters
    ----------
    obj
        array of data
    buffer_rows
        buffer to concatenate with.
    idx
        integer index used control the behaviour for the first and last chunk.
    max_idx
        largest index.
    delete_input
        flag to delete the input files.

    Returns
    -------
    (obj,buffer,mode) tuple of the data, the buffer for the next chunk and the IO mode for file writing.
    """
    counts = ak.run_lengths(ak_obj.evtid)
    rows = ak.num(ak_obj, axis=-1)
    end_rows = counts[-1]

    if idx == 0:
        mode = "of" if (delete_input) else "append"
        obj = ak_obj[0 : rows - end_rows]
        buffer_rows = copy.deepcopy(ak_obj[rows - end_rows :])
    elif idx != max_idx:
        mode = "append"
        obj = ak.concatenate((buffer_rows, ak_obj[0 : rows - end_rows]))
        buffer_rows = copy.deepcopy(ak_obj[rows - end_rows :])
    else:
        mode = "append"
        obj = ak.concatenate((buffer_rows, ak_obj))

    return obj, buffer_rows, mode
