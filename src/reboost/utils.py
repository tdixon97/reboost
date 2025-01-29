from __future__ import annotations

import importlib
import json
import logging
import re
from contextlib import contextmanager
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def load_dict(fname: str, ftype: str | None = None) -> dict:
    """Load a text file as a Python dict.

    Parameters
    ----------
    fname
        path to the file
    ftype
        type of file, can be either json or yaml.
    """
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


def _search_string(string: str):
    """Capture the characters matching the pattern for a function call."""

    pattern = r"\b([a-zA-Z_][a-zA-Z0-9_\.]*)\s*\("
    return re.findall(pattern, string)


def get_function_string(expr: str, aliases: dict | None = None) -> tuple[str, dict]:
    """Get a function call to evaluate.

    Search for any patterns matching the pattern for a function call.
    We also detect any cases of aliases being used, by default
    just for `numpy` as `np` and `awkward` as `ak`. In this
    case, the full name is replaces with the alias in the expression
    and also in the output globals dictionary.

    It is possible to chain together functions eg:

    .. code-block:: python

        ak.num(np.array([1, 2]))

    and all packages will be imported.

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

    funcs = _search_string(expr.strip())
    for func_call in funcs:
        # no "." then can't be a module
        if "." not in func_call:
            continue

        subpackage, func = func_call.rsplit(".", 1)
        package = subpackage.split(".")[0]

        # import the subpackage
        for name, short_name in aliases.items():
            subpackage = subpackage.replace(short_name, name)

        # handle the aliases
        package_import = package
        for name, short_name in aliases.items():
            if package == short_name:
                package_import = name

        # build globals
        try:
            importlib.import_module(subpackage, package=__package__)

            globs = globs | {
                package: importlib.import_module(package_import),
            }
        except Exception:
            msg = f"Function {package_import} cannot be imported"
            log.debug(msg)
            continue

    return expr, globs


def merge_dicts(dict_list: list) -> dict:
    """Merge a list of dictionaries, concatenating the items where they exist.

    Parameters
    ----------
    dict_list
        list of dictionaries to merge

    Returns
    -------
    a new dictionary after merging.

    Examples
    --------

    >>> merge_dicts([{"a":[1,2,3],"b":[2]},{"a":[4,5,6],"c":[2]}])
    {"a":[1,2,3,4,5,6],"b":[2],"c":[2]}

    """
    merged = {}

    for tmp_dict in dict_list:
        for key, item in tmp_dict.items():
            if key in merged:
                merged[key].extend(item)
            else:
                merged[key] = item

    return merged


@contextmanager
def filter_logging(level):
    logger = logging.getLogger("root")
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)
