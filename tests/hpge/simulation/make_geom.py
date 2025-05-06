from __future__ import annotations

import legendhpges as hpges
import numpy as np
import pyg4ometry as pg4
from legendtestdata import LegendTestData

ldata = LegendTestData()

reg = pg4.geant4.Registry()

natge = hpges.materials.make_natural_germanium(reg)
hpge_l = hpges.make_hpge(
    ldata["legend/metadata/hardware/detectors/germanium/diodes/V99000A.yaml"],
    name="V99000A",
    registry=reg,
    material=natge,
)

# create a world volume
world_s = pg4.geant4.solid.Orb("World_s", 20, registry=reg, lunit="cm")
world_l = pg4.geant4.LogicalVolume(world_s, "G4_Galactic", "World", registry=reg)
reg.setWorld(world_l)

pg4.geant4.PhysicalVolume([0, 0, 0], [0, 0, 0, "m"], hpge_l, "V99000A", world_l, registry=reg)

# finally create a small radioactive source
source_s = pg4.geant4.solid.Tubs("Source_s", 0, 1, 1, 0, 2 * np.pi, registry=reg)
source_l = pg4.geant4.LogicalVolume(source_s, "G4_BRAIN_ICRP", "Source_L", registry=reg)
pg4.geant4.PhysicalVolume([0, 0, 0], [0, 0, -1, "cm"], source_l, "Source", world_l, registry=reg)

# viewer = pg4.visualisation.VtkViewerColoured()
# viewer.addLogicalVolume(reg.getWorldVolume())
# viewer.view()

w = pg4.gdml.Writer()
w.addDetector(reg)
w.write("geometry.gdml")
