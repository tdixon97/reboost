from __future__ import annotations

import importlib
import itertools
import logging
import re
import time
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from pathlib import Path

from dbetto import AttrsDict
from lgdo import lh5
from lgdo.types import Struct, Table, VectorOfVectors

from .profile import ProfileDict

log = logging.getLogger(__name__)


def get_table_names(tcm: VectorOfVectors) -> dict:
    """Extract table names from tcm.attrs['tables'] and return them as a dictionary."""
    raw = tcm.attrs["tables"]
    cleaned = raw.strip("[]").replace(" ", "").replace("'", "")
    tables = cleaned.split(",")
    tables = [tab.split("/")[-1] for tab in tables]

    return {name: idx for idx, name in enumerate(tables)}


def get_wo_mode(
    group: int, out_det: int, in_det: int, chunk: int, new_hit_file: bool, overwrite: bool = False
) -> str:
    """Get the mode for lh5 file writing.

    If all indices are 0 and we are writing a new output file
    then the mode "overwrite_file" is used (if the overwrite) flag
    is set, otherwise the mode "write_safe" is used.

    Otherwise the code choses between "append_column" if this is the
    first time a group is being written to the file, or "append"

    Parameters
    ----------
    group
        the index of the processing group
    out_det
        the index of the output detector
    in_det
        the index of the input detector
    chunk
        the chunk index
    new_hit_file
        a flag of whether we are writing a new hit file
    overwrite
        a flag of whether to overwrite the old file.

    Returns
    -------
    the mode for IO
    """
    indices = [group, out_det, in_det, chunk]

    good_idx = all(i == 0 for i in indices)

    if good_idx and new_hit_file:
        return "overwrite_file" if overwrite else "write_safe"

    # if we have a detector not the first and chunk 0 append column
    is_ac = ((in_det > 0) or (out_det > 0)) & (chunk == 0)
    is_ac = is_ac or (in_det == 0 and out_det == 0 and chunk == 0 and (group > 0))

    if is_ac and new_hit_file:
        return "append_column"
    return "append"


def get_file_dict(
    stp_files: list[str] | str,
    glm_files: list[str] | str | None,
    hit_files: list[str] | str | None = None,
) -> AttrsDict:
    """Get the file info as a AttrsDict.

    Creates an :class:`dbetto.AttrsDict` with keys `stp_files`,
    `glm_files` and `hit_files`. Each key contains a list of
    file-paths (or `None`).

    Parameters
    ----------
    stp_files
        string or list of strings of the stp files.
    glm_files
        string or list of strings of the glm files, or None in which
        case the glm will be created in memory.
    hit_files
        string or list of strings of the hit files, if None the output
        files will be created in memory.
    """
    # make a list of the right length
    if isinstance(stp_files, str):
        stp_files = [stp_files]

    glm_files_list = [None] * len(stp_files) if glm_files is None else glm_files

    # make a list of files in case
    # 1) hit_files is a str and stp_files is a list
    # 2) hit_files and stp_files are both lists of different length

    hit_is_list = isinstance(hit_files, list)

    if not hit_is_list:
        hit_files_list = [hit_files] * len(stp_files)
    elif hit_is_list and len(hit_files) == 1 and len(stp_files) > 1:
        hit_files_list = [hit_files[0]] * len(stp_files)
    else:
        hit_files_list = hit_files

    files = {}

    for file_type, file_list in zip(
        ["stp", "glm", "hit"], [stp_files, glm_files_list, hit_files_list]
    ):
        if isinstance(file_list, str):
            files[file_type] = [file_list]
        else:
            files[file_type] = file_list

    return AttrsDict(files)


def get_file_list(path: str | None, threads: int | None = None) -> list[str]:
    """Get a list of files accounting for the multithread index."""
    if threads is None or path is None:
        return path
    return [f"{(Path(path).with_suffix(''))}_t{idx}.lh5" for idx in range(threads)]


def copy_units(tab: Table) -> dict:
    """Extract a dictionary of attributes (i.e. units).

    Parameters
    ----------
    tab
        Table to get the units from.

    Returns
    -------
    a dictionary with the units for each field
    in the table.
    """
    units = {}

    for field in tab:
        if "units" in tab[field].attrs:
            units[field] = tab[field].attrs["units"]

    return units


def assign_units(tab: Table, units: Mapping) -> Table:
    """Copy the attributes from the map of attributes to the table.

    Parameters
    ----------
    tab
        Table to add attributes to.
    units
        mapping (dictionary like) of units of each field

    Returns
    -------
    an updated table with LGDO attributes.
    """
    for field in tab:
        if field in units:
            if not isinstance(tab[field], VectorOfVectors):
                tab[field].attrs["units"] = units[field]
            else:
                tab[field].flattened_data.attrs["units"] = units[field]

    return tab


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
    ----------
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


def get_channels_from_groups(names: list | str | None, groupings: dict | None = None) -> list:
    """Get a list of channels from a list of groups.

    Parameters
    ----------
    names
        list of channel names
    groupings
        dictionary of the groupings of channels

    Returns
    -------
    list of channels
    """
    if names is None:
        channels_e = []
    elif isinstance(names, str):
        channels_e = groupings[names]
    elif isinstance(names, list):
        channels_e = list(itertools.chain.from_iterable([groupings[e] for e in names]))
    else:
        msg = f"names {names} must be list or str or `None`"
        raise ValueError(msg)

    return channels_e


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


def _check_input_file(parser, file: str | Iterable[str], descr: str = "input") -> None:
    file = (file,) if isinstance(file, str) else file
    not_existing = [f for f in file if not Path(f).exists()]
    if not_existing != []:
        parser.error(f"{descr} file(s) {''.join(not_existing)} missing")


def _check_output_file(parser, file: str | Iterable[str] | None, optional: bool = False) -> None:
    if file is None and optional:
        return

    file = (file,) if isinstance(file, str) else file
    for f in file:
        if Path(f).exists():
            parser.error(f"output file {f} already exists")


def write_lh5(
    hit_table: Table,
    file: str,
    time_dict: ProfileDict,
    out_field: str,
    out_detector: str,
    wo_mode: str,
):
    """Write the lh5 file. This function handles writing first the data as a struct and then appending to this.

    Parameters
    ----------
    hit_table
        the table to write
    file
        the file to write to
    time_dict
        the dictionary of timing information to update.
    out_field
        output field
    out_detector
        output detector name
    wo_mode
        the mode to pass to `lh5.write`
    """
    if time_dict is not None:
        start_time = time.time()

    if wo_mode not in ("a", "append"):
        lh5.write(
            Struct({out_detector: hit_table}),
            out_field,
            file,
            wo_mode=wo_mode,
        )
    else:
        lh5.write(
            hit_table,
            f"{out_field}/{out_detector}",
            file,
            wo_mode=wo_mode,
        )
    if time_dict is not None:
        time_dict.update_field("write", start_time)
