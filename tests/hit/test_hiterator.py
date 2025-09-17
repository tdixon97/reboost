from __future__ import annotations

import copy
from pathlib import Path

import awkward as ak
import legendhpges
import lgdo
import numpy as np
import pyg4ometry
import pygeomtools
import pytest
from dbetto import AttrsDict
from lgdo import Array, Table, VectorOfVectors, lh5

from reboost import Hiterator
from reboost.hpge import surface
from reboost.math import functions, stats

configs = Path(__file__).parent / "configs"


@pytest.fixture(scope="module")
def test_gen_lh5(tmptestdir):
    # write a basic lh5 file

    stp_path = str(tmptestdir / "basic.lh5")

    data = {}
    data["evtid"] = Array([0, 1])
    data["edep"] = VectorOfVectors([[100, 200], [10, 20, 300]])  # keV
    data["time"] = VectorOfVectors([[0, 1.5], [0.1, 2.1, 3.7]])  # ns

    data["xloc"] = VectorOfVectors([[0.01, 0.02], [0.001, 0.003, 0.005]])  # m
    data["yloc"] = VectorOfVectors([[0.01, 0.02], [0.001, 0.003, 0.005]])  # m
    data["zloc"] = VectorOfVectors([[0.04, 0.02], [0.001, 0.023, 0.005]])  # m
    data["dist_to_surf"] = VectorOfVectors([[0.04, 0.02], [0.011, 0.003, 0.051]])  # m

    vertices = [0, 1]
    tab = Table(data)
    tab2 = copy.deepcopy(tab)

    lh5.write(tab, "stp/det1", stp_path, wo_mode="of")
    lh5.write(tab2, "stp/det2", stp_path, wo_mode="append")
    lh5.write(
        Table({"evtid": Array(vertices)}),
        "vtx",
        stp_path,
        wo_mode="append",
    )

    return stp_path


def test_full_chain(test_gen_lh5, tmptestdir):
    detectors = ["det1", "det2"]

    pars = AttrsDict(
        {
            "det1": {"fccd_in_mm": 0.71, "dlf": 0.7, "reso_in_sigma": 1.1, "name": "BEGe"},
            "det2": {"fccd_in_mm": 2.2, "dlf": 0.4, "reso_in_sigma": 2.32, "name": "Coax"},
        }
    )

    geometry = pyg4ometry.gdml.Reader(str(configs / "geom.gdml")).getRegistry()
    registry = pyg4ometry.geant4.Registry()

    metadata = AttrsDict(
        {k: pygeomtools.get_sensvol_metadata(geometry, pars[k].name) for k in detectors}
    )

    for det in detectors:
        it = Hiterator(
            input_files=test_gen_lh5,
            glm_files=None,
            input_detector=det,
            output_detectors=det,
            input_hdf5_group="stp",
            buffer_size_rows=5_000_000,
            overwrite=True,
        )

        # get some objects
        hpge = legendhpges.make_hpge(metadata[det], registry=registry)
        name = pars[det].name

        for chunk in it:
            _t0 = ak.fill_none(ak.firsts(chunk.data.time, axis=-1), np.nan)

            _distance = surface.distance_to_surface(
                chunk.data.xloc,
                chunk.data.yloc,
                chunk.data.zloc,
                hpge,
                geometry.physicalVolumeDict[name].position.eval(),
                distances_precompute=chunk.data.dist_to_surf,
                precompute_cutoff=0.002,
                surface_type="nplus",
            )

            _activeness = functions.piecewise_linear_activeness(
                _distance, fccd=pars[chunk.in_detector].fccd_in_mm, dlf=pars[chunk.in_detector].dlf
            ).view_as("ak")

            _active_e = ak.sum(chunk.data.edep.view_as("ak") * _activeness, axis=-1)

            _smeared_e = stats.gaussian_sample(_active_e, pars[chunk.in_detector].reso_in_sigma)

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
