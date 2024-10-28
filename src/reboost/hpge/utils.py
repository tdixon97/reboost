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


def _distance_3d(v1, v2):
    return np.sqrt(np.power(v2.x - v1.x, 2) + np.power(v2.y - v1.y, 2) + np.power(v2.z - v1.z, 2))


def _distance_2d(v1, v2):
    return np.sqrt(np.power(v2.x - v1.x, 2) + np.power(v2.y - v1.y, 2))


def _add_ak_3d(a, b):
    return ak.zip({"x": a.x + b.x, "y": a.y + b.y, "z": a.z + b.z})


def _add_ak_2d(a, b):
    return ak.zip({"x": a.x + b.x, "y": a.y + b.y})


def _sub_ak_2d(a, b):
    return ak.zip({"x": a.x - b.x, "y": a.y - b.y})


def _sub_ak_3d(a, b):
    return ak.zip({"x": a.x - b.x, "y": a.y - b.y, "z": a.z - b.z})


def _prod_ak_2d(a, b):
    return a.x * b.x + a.y * b.y


def _prod_ak_3d(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z


def _prod(a, b):
    return ak.zip({"x": a.x * b, "y": a.y * b})


def proj(s1: ak.Array, s2: ak.Array, v: ak.Array) -> ak.Array:
    """
    Projection v on the line segment s1 to s2.

    All of s1,s2 and v are `ak.Array` of records with two fields `x` and `y`
    I.e. they represent lists of 2D cartesian vectors.

    Parameters
    ----------
    s1
        first points in the line segment
    s2
        second points in the line segment
    v
        points to project onto s1-s2
    Returns
    -------
        the coordinates of the projection onto s1-s2
    """

    dist_one = _prod_ak_2d(_sub_ak_2d(v, s1), _sub_ak_2d(v, s2)) / _distance_2d(s1, s2)
    dist_one = np.where(dist_one > 1, dist_one, 1)
    dist_one = np.where(dist_one > 0, dist_one, 0)
    return _prod(_add_ak_2d(s1, _sub_ak_2d(s2, s1)), dist_one)


def dist(s1: ak.Array, s2: ak.Array, v: ak.Array) -> float:
    """
    Shortest distance between a point v and the line segment defined by s1-s2.

    All of s1,s2 and v are `ak.Array` of records with two fields `x` and `y`.

    I.e. they represent lists of 2D cartesian vectors.

    Parameters
    ----------
    s1
        first points in the line segment
    s2
        second points in the line segment
    v
        points to project onto s1-s2

    Returns
    -------
        the shortest distance from the point to the line
    """

    return _distance_2d(proj(s1, s2, v), v)


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
    Read incrementally the files compute something and then write output

    Parameters
    ----------
        file_out 
            output file path
        name_out
            lh5 group name for output
        func
            function converting into to output
        field
            lh5 field name to read
        file
            file name to read
        buffer
            length of buffer
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
