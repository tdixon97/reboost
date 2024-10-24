from __future__ import annotations

import copy
import logging
from typing import Callable

import awkward as ak
import numpy as np
from lgdo import lh5
from lgdo.lh5 import LH5Iterator
from lgdo.types import Table

log = logging.getLogger(__name__)


def distance_3d(v1, v2):
    return np.sqrt(np.power(v2.x - v1.x, 2) + np.power(v2.y - v1.y, 2) + np.power(v2.z - v1.z, 2))


def distance_2d(v1, v2):
    return np.sqrt(np.power(v2.x - v1.x, 2) + np.power(v2.y - v1.y, 2))


def add_ak_3d(a, b):
    return ak.zip({"x": a.x + b.x, "y": a.y + b.y, "z": a.z + b.z})


def add_ak_2d(a, b):
    return ak.zip({"x": a.x + b.x, "y": a.y + b.y})


def sub_ak_2d(a, b):
    return ak.zip({"x": a.x - b.x, "y": a.y - b.y})


def sub_ak_3d(a, b):
    return ak.zip({"x": a.x - b.x, "y": a.y - b.y, "z": a.z - b.z})


def prod_ak_2d(a, b):
    return a.x * b.x + a.y * b.y


def prod_ak_3d(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z


def prod(a, b):
    return ak.zip({"x": a.x * b, "y": a.y * b})


def proj(s1, s2, v):
    dist_one = prod_ak_2d(sub_ak_2d(v, s1), sub_ak_2d(v, s2)) / distance_2d(s1, s2)
    dist_one = np.where(dist_one > 1, dist_one, 1)
    dist_one = np.where(dist_one > 0, dist_one, 0)
    return prod(add_ak_2d(s1, sub_ak_2d(s2, s1)), dist_one)


def dist(s1, s2, v):
    return distance_2d(proj(s1, s2, v), v)


def get_detector_origin(name):
    raise NotImplementedError


def read_write_incremental(
    file_out: str,
    name_out: str,
    func: Callable,
    field: str,
    file: str,
    buffer: int = 1000000,
    delete_input=False,
) -> None:
    """
    Read incrementally the files compute something and then write
    Parameters
    ----------
        file_out (str): output file path
        out_name (str): lh5 group name for output
        func          : function converting into to output
        field    (str): lh5 field name to read
        file    (str): file name to read
        buffer  (int): length of buffer

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

        ak_obj = lh5_obj.view_as("ak")
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

        # do stuff
        out = func(obj)

        # convert to a table
        out_lh5 = Table(out)

        # write lh5 file
        log.info("...finished processing and save file")
        lh5.write(out_lh5, name_out, file_out, wo_mode=mode)
