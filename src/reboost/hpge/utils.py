from __future__ import annotations

import copy
import importlib
import json
import logging
import re
from collections import namedtuple
from pathlib import Path

import awkward as ak
import legendhpges
import pyg4ometry
import yaml

log = logging.getLogger(__name__)

reg = pyg4ometry.geant4.Registry()


def get_hpge(meta_path: str, pars: dict, detector: str) -> legendhpges.HPGe:
    """Extract the :class:`legendhpges.HPGe` object from metadata.

    Parameters
    ----------
    meta_path
        path to the folder with the `diodes` metadata.
    pars
        dictionary of parameters.
    detector
        remage output name for the detector

    Returns
    -------
    hpge
        the `legendhpges` object for the detector.
    """

    meta_name = pars.get("meta_name", f"{detector}.json")
    meta_dict = Path(meta_path) / Path(meta_name)
    return legendhpges.make_hpge(meta_dict, registry=reg)


def get_phy_vol(
    reg: pyg4ometry.geant4.Registry | None, pars: dict, detector: str
) -> pyg4ometry.geant4.PhysicalVolume:
    """Extract the :class:`pyg4ometry.geant4.PhysicalVolume` object from GDML

    Parameters
    ----------
    reg
        Geant4 registry from GDML
    pars
        dictionary of parameters.
    detector
        remage output name for the detector.

    Returns
    -------
    phy_vol
        the `pyg4ometry.geant4.PhysicalVolume` object for the detector
    """
    if reg is not None:
        phy_name = pars.get("phy_vol_name", f"{detector}")
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
