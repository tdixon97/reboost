from __future__ import annotations

import logging

import awkward as ak
import numpy as np
from lgdo import Array

from ..optmap import convolve

log = logging.getLogger(__name__)


def load_optmap(map_file):
    return convolve.open_optmap(map_file)


def detected_photoelectrons(
    edep, evtid, particle, time, xloc, yloc, zloc, optmap, material, spm_detector
) -> Array:
    """R90 HPGe pulse shape heuristic.

    Parameters
    ----------
    hits
        the scintillator hits table.
    optmap
        the optical map loaded via py:func:`load_optmap`
    material
        scintillating material name
    spm_detector
        SiPM
    """
    hits = ak.Array(
        {
            "edep": edep,
            "evtid": evtid,
            "particle": particle,
            "time": time,
            "xloc": xloc,
            "yloc": yloc,
            "zloc": zloc,
        }
    )
    edep_df = convolve._reflatten_scint_vov(hits).to_numpy()

    scint_mat_params = convolve._get_scint_params(material)
    output_map = convolve.iterate_stepwise_depositions_pois(
        edep_df, optmap, scint_mat_params, mode="no-fano", det_uid=spm_detector
    )

    return ak.Array(np.ones(shape=3))

    raise ValueError(output_map)
    msg = "already here?"
    raise RuntimeError(msg)

    # ph_count_o, tbl = get_output_table(output_map)
    # log.debug("output photons: %d energy depositions -> %d photons", len(output_map), ph_count_o)

    msg = "already here?"
    raise RuntimeError(msg)
