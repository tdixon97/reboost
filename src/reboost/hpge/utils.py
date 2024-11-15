from __future__ import annotations

import copy
import importlib
import json
import logging
import re
from collections import namedtuple
from pathlib import Path
from typing import NamedTuple

import awkward as ak
import legendhpges
import numpy as np
import pyg4ometry
import yaml
from lgdo import lh5
from numpy.typing import ArrayLike, NDArray

log = logging.getLogger(__name__)

reg = pyg4ometry.geant4.Registry()


class FileInfo(NamedTuple):
    """NamedTuple storing the information on the input files"""

    file_list: list[str]
    """list of strings of the selected files."""

    file_indices: list[int]
    """list of integers of the indices of the files."""

    file_start_global_evtids: list[int]
    """list of integers of the first global evtid for each file."""

    first_global_evtid: int
    """first global evtid to process."""

    last_global_evtid: int
    """Last global evtid to process."""


def get_selected_files(
    file_list: list[str], table: str, n_evtid: int | None, start_evtid: int
) -> FileInfo:
    """Get the files to read based on removing those with global evtid out of the selected range.

    - expands wildcards,
    - extracts number of evtid per file,
    - removes files outside of the range `low_evtid` to `high_evtid`.

    Parameters
    ----------
    file_list
        list of files to process (can include wildcards)
    table
        lh5 input field
    n_evtid
        number of simulation events to process.
    high_global_evtid
        start_evtid: first (global) simulation event to process.

    Returns
    -------
    `FileInfo` object with information on the files.
    """
    # expand wildcards
    expanded_list_file_in = get_file_list(path=file_list)

    n_sim = get_num_simulated(expanded_list_file_in, table=table)

    # first global index of each file
    cum_nsim = np.concatenate([[0], np.cumsum(n_sim)])

    low_global_evtid, high_global_evtid = get_global_evtid_range(
        start_evtid=start_evtid, n_evtid=n_evtid, n_tot_sim=cum_nsim[-1]
    )
    file_indices = np.array(
        get_files_to_read(
            cum_nsim, start_glob_evtid=low_global_evtid, end_glob_evtid=high_global_evtid
        )
    )

    # select just the necessary files
    file_list_sel = np.array(expanded_list_file_in)[file_indices]
    start_evtid_sel = cum_nsim[file_indices]

    return FileInfo(
        file_list_sel, file_indices, start_evtid_sel, low_global_evtid, high_global_evtid
    )


def get_num_simulated(file_list: list, table: str = "hit") -> int:
    """Loop over a list of files and extract the number of simulated events.

    Based on the size of the `vertices` tables.

    Parameters
    ----------
    file_list
        list of input files (each must contain the vertices table)
    table
        name of the lh5 input table.
    """
    n = []
    for file in file_list:
        it = lh5.LH5Iterator(file, f"{table}/vertices", buffer_len=int(5e6))
        n.append(it._get_file_cumlen(0))

    msg = f"files contain {n} events"
    log.info(msg)
    return n


def get_file_list(path: str | list[str]) -> NDArray:
    """Get list of files to read.

    Parameters
    ----------
    path
        either a string or a list of strings containing files paths to process. May contain wildcards which are expanded.

    Returns
    -------
    sorted array of files, after expanding wildcards, removing duplicates and sorting.

    """

    if isinstance(path, str):
        path = [path]

    path_out_list = []

    for ptmp in path:
        ptmp_path = Path(ptmp)
        dir_tmp = ptmp_path.parent
        pattern_tmp = ptmp_path.name

        path_out_list.extend(dir_tmp.glob(pattern_tmp))
    path_out_list_str = [str(ptmp) for ptmp in path_out_list]
    return np.array(np.sort(np.unique(path_out_list_str)))


def get_global_evtid_range(
    start_evtid: int, n_evtid: int | None, n_tot_sim: int
) -> tuple[int, int]:
    """Get the global evtid range"""

    # some checks
    if (n_evtid is not None) and (start_evtid + n_evtid > n_tot_sim):
        msg = "Index are out of the range of the simulation."
        raise ValueError(msg)

    start_glob_index = start_evtid
    end_glob_index = start_evtid + n_evtid - 1 if (n_evtid is not None) else n_tot_sim - 1
    return start_glob_index, end_glob_index


def get_files_to_read(cum_n_sim: ArrayLike, start_glob_evtid: int, end_glob_evtid: int) -> NDArray:
    """Get the index of files to read based on the number of evtid to read and the start evtid.

    Parameters
    ----------
    cum_n_sim
        cumulative list of the number of evtid per file.
    start_glob_evtid
        first global evtid to include.
    end_glob_evtid
        last global evtid to process.

    Returns
    -------
    array of the indices of files to process.
    """
    # find which files to read

    file_indices = []
    cum_n_sim = np.array(cum_n_sim)

    for i, (low, high) in enumerate(zip(cum_n_sim, cum_n_sim[1:] - 1)):
        if (high >= start_glob_evtid) & (low <= end_glob_evtid):
            file_indices.append(i)
    return np.array(file_indices)


def get_include_chunk(
    global_evtid: ak.Array,
    start_glob_evtid: int,
    end_glob_evtid: int,
) -> bool:
    """Check if a chunk can be skipped based on evtid range.

    Parameters
    ----------
    global_evtid
        awkward array of the (local) evtid in the chunk
    start_glob_evtid
        first global evtid to include.
    end_glob_evtid
        last global evtid to process.
    Returns
    -------
    boolean flag of whether to include in the chunk.

    """
    low = ak.min(global_evtid, axis=0)
    high = ak.max(global_evtid, axis=0)
    return (high >= start_glob_evtid) & (low <= end_glob_evtid)


def get_hpge(meta_path: str | None, pars: NamedTuple, detector: str) -> legendhpges.HPGe:
    """Extract the :class:`legendhpges.HPGe` object from metadata.

    Parameters
    ----------
    meta_path
        path to the folder with the `diodes` metadata.
    pars
        named tuple of parameters.
    detector
        remage output name for the detector

    Returns
    -------
    hpge
        the `legendhpges` object for the detector.
    """
    reg = pyg4ometry.geant4.Registry()
    if meta_path is not None:
        meta_name = pars.meta_name if ("meta_name" in pars._fields) else f"{detector}.json"
        meta_dict = Path(meta_path) / Path(meta_name)
        return legendhpges.make_hpge(meta_dict, registry=reg)
    return None


def get_phy_vol(
    reg: pyg4ometry.geant4.Registry | None, pars: NamedTuple, detector: str
) -> pyg4ometry.geant4.PhysicalVolume:
    """Extract the :class:`pyg4ometry.geant4.PhysicalVolume` object from GDML

    Parameters
    ----------
    reg
        Geant4 registry from GDML
    pars
        named tuple of parameters.
    detector
        remage output name for the detector.

    Returns
    -------
    phy_vol
        the `pyg4ometry.geant4.PhysicalVolume` object for the detector
    """
    if reg is not None:
        phy_name = pars.phy_vol_name if ("phy_vol_name" in pars._fields) else f"{detector}"
        return reg.physicalVolumeDict[phy_name]
    return None


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
    ak_obj: ak.Array,
    buffer_rows: ak.Array | None,
    idx: int,
    max_idx: int,
    delete_input: bool = False,
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
        buffer_rows = None

    return obj, buffer_rows, mode


__file_extensions__ = {"json": [".json"], "yaml": [".yaml", ".yml"]}


def load_dict(fname: str, ftype: str | None = None) -> dict:
    """Load a text file as a Python dict."""
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
