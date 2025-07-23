from __future__ import annotations

import logging

import awkward as ak
from lgdo import VectorOfVectors

from ..optmap import convolve

log = logging.getLogger(__name__)


def load_optmap_all(map_file: str) -> convolve.OptmapForConvolve:
    """Load an optical map file for later use with :py:func:`detected_photoelectrons`."""
    return convolve.open_optmap(map_file)


def load_optmap(map_file: str, spm_det_uid: int) -> convolve.OptmapForConvolve:
    """Load an optical map file for later use with :py:func:`detected_photoelectrons`."""
    return convolve.open_optmap_single(map_file, spm_det_uid)


def detected_photoelectrons(
    edep: ak.Array,
    evtid: ak.Array,
    particle: ak.Array,
    time: ak.Array,
    xloc: ak.Array,
    yloc: ak.Array,
    zloc: ak.Array,
    optmap: convolve.OptmapForConvolve,
    material: str,
    spm_detector_uid: int,
) -> VectorOfVectors:
    """Derive the number of detected photoelectrons (p.e.) from scintillator hits using an optical map.

    Parameters
    ----------
    optmap
        the optical map loaded via py:func:`load_optmap`
    material
        scintillating material name
    spm_detector
        SiPM detector uid as used in the optical map.
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

    scint_mat_params = convolve._get_scint_params(material)
    pe = convolve.iterate_stepwise_depositions_pois(
        hits, optmap, scint_mat_params, mode="no-fano", det_uid=spm_detector_uid
    )

    return VectorOfVectors(pe)
