from __future__ import annotations

from pathlib import Path

import awkward as ak
import legendhpges
import lgdo
import numpy as np
import pyg4ometry
import pygeomtools
from dbetto import AttrsDict
from lgdo import lh5

import reboost
from reboost import Hiterator

configs = Path(__file__).parent / "configs"


def test_full_chain(test_gen_lh5, tmptestdir):
    detectors = ["det1", "det2"]

    pars = AttrsDict(
        {
            "det1": {
                "fccd_in_mm": 0.71,
                "dlf": 0.7,
                "reso_in_sigma": 1.1,
            },
            "det2": {
                "fccd_in_mm": 2.2,
                "dlf": 0.4,
                "reso_in_sigma": 2.32,
            },
        }
    )

    geometry = pyg4ometry.gdml.Reader(str(configs / "geom.gdml")).getRegistry()
    registry = pyg4ometry.geant4.Registry()

    metadata = AttrsDict({k: pygeomtools.get_sensvol_metadata(geometry, k) for k in detectors})

    hpges = {k: legendhpges.make_hpge(metadata[k], registry=registry) for k in detectors}

    it = Hiterator(
        input_files=test_gen_lh5,
        glm_files=None,
        detectors_mapping={"output": detectors},
        input_hdf5_group="stp",
        buffer_size_rows=5_000_000,
        overwrite=True,
    )

    for chunk in it:
        _t0 = ak.fill_none(ak.firsts(chunk.data.time, axis=-1), np.nan)

        _distance = reboost.hpge.surface.distance_to_surface(
            chunk.data.xloc,
            chunk.data.yloc,
            chunk.data.zloc,
            hpges[chunk.in_detector],
            geometry.physicalVolumeDict[chunk.in_detector].position.eval(),
            distances_precompute=chunk.data.dist_to_surf,
            precompute_cutoff=0.002,
            surface_type="nplus",
        )

        _activeness = reboost.math.functions.piecewise_linear_activeness(
            _distance, fccd=pars[chunk.in_detector].fccd_in_mm, dlf=pars[chunk.in_detector].dlf
        )

        _active_e = ak.sum(chunk.data.edep * _activeness, axis=-1)

        _smeared_e = reboost.math.stats.gaussian_sample(
            chunk.data.active_energy, pars[chunk.in_detector].reso_in_sigma
        )

        table = lgdo.Table()
        table.add_field("t0", lgdo.Array(_t0))
        table.add_field("distance_to_nplus", lgdo.VectorOfVectors(_distance))
        table.add_field("activeness", lgdo.VectorOfVectors(_activeness))
        table.add_field("active_energy", lgdo.Array(_active_e))
        table.add_field("smeared_energy", lgdo.Array(_smeared_e))

        lh5.write(
            table,
            f"hit/{chunk.out_detector}",
            tmptestdir / "beta-small-hiterator.lh5",
        )
