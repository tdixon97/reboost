from __future__ import annotations

import importlib
import json
import logging
import re
from collections import namedtuple
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def load_dict(fname: str, ftype: str | None = None) -> dict:
    """Load a text file as a Python dict."""
    __file_extensions__ = {"json": [".json"], "yaml": [".yaml", ".yml"]}

    fname = Path(fname)

    # determine file type from extension
    if ftype is None:
        for _ftype, exts in __file_extensions__.items():
            if fname.suffix in exts:
                ftype = _ftype

    msg = f"loading {ftype} dict from: {fname}"
    log.debug(msg)

    with fname.open() as f:
        if ftype == "json":
            return json.load(f)
        if ftype == "yaml":
            return yaml.safe_load(f)

        msg = f"unsupported file format {ftype}"
        raise NotImplementedError(msg)


def get_wo_mode(
    file_idx: int,
    group_idx: int,
    in_det_idx: int,
    out_det_idx: int,
    chunk_idx: int,
    max_chunk_idx: int,
) -> str:
    raise NotImplementedError


def get_function_string(expr: str, aliases: dict | None = None) -> tuple[str, dict]:
    """Get a function call to evaluate.

    Based on the expression to evaluate we search for any
    regular expression matching the pattern for a function call,
    i.e. parentheses. The module is captured and imported.
    The function returns a dictionary of modules.

    We also detect any cases of aliases being used, by default
    just for `numpy` as `np` and `awkward` as `ak`. In this
    case, the full name is replaces with the alias in the expression
    and also in the output globals dictionary.

    It is possible to chain together functions eg:

    .. code-block:: python

        ak.num(np.array([1, 2]))

    and all packages will be imported. No other operations are supported.

    Parameters
    ---------
    expr
        expression to evaluate.
    aliases
        dictionary of package aliases for names used in dictionary. These allow to
        give shorter names to packages. This is combined with two defaults `ak` for
        `awkward` and `np` for `numpy`. If `None` is supplied only these are used.

    Returns
    -------
    a tuple of call string and dictionary of the imported global packages.
    """

    # aliases for easier lookup
    aliases = (
        {"numpy": "np", "awkward": "ak"}
        if aliases is None
        else aliases | {"numpy": "np", "awkward": "ak"}
    )

    # move to only alias names
    for name, short_name in aliases.items():
        expr = expr.replace(name, short_name)

    globs = {}
    # search on the whole expression
    args_str = re.search(r"\((.*)\)$", expr.strip()).group(1)
    expr_tmp = expr

    # recursively search through
    while args_str is not None:
        # get module and function names
        func_call = expr_tmp.strip().split("(")[0]
        subpackage, func = func_call.rsplit(".", 1)
        package = subpackage.split(".")[0]

        # import the subpackage
        for name, short_name in aliases.items():
            subpackage = subpackage.replace(short_name, name)
        importlib.import_module(subpackage, package=__package__)

        # handle the aliases
        package_import = package
        for name, short_name in aliases.items():
            if package == short_name:
                package_import = name

        # build globals
        globs = globs | {
            package: importlib.import_module(package_import),
        }
        # search for a function pattern with the arguments

        patterns = re.search(r"\((.*)\)$", args_str.strip())
        expr_tmp = args_str

        args_str = patterns.group(1) if patterns is not None else None

    return expr, globs


def dict2tuple(dictionary: dict) -> namedtuple:
    return namedtuple("parameters", dictionary.keys())(**dictionary)
