(optics)=

# Using optical map

:::{warning}

Work in progress, more will be added later! _reboost-optical_ is also not integrated with
the remaining _reboost_ stack.

:::

## Creating optical maps

### 1. Running remage simulations to get stp file

The map generation is performed directly with _remage_(_reboost_ is not involved in this
step). An example macro to showcase the required settings (`map.mac`):

```
/RMG/Processes/OpticalPhysics true
/RMG/Processes/OpticalPhysicsMaxOneWLSPhoton true

/RMG/Geometry/RegisterDetectorsFromGDML Optical

/RMG/Geometry/GDMLDisableOverlapCheck true

/run/initialize

/RMG/Output/NtuplePerDetector false
/RMG/Output/Vertex/StorePrimaryParticleInformation false
/RMG/Output/Vertex/StoreSinglePrecisionEnergy true
/RMG/Output/Vertex/StoreSinglePrecisionPosition true

# confinement in disk around array
/RMG/Generator/Confine Volume

/RMG/Generator/Confinement/SamplingMode IntersectPhysicalWithGeometrical
/RMG/Generator/Confinement/ForceContainmentCheck true

/RMG/Generator/Confinement/Physical/AddVolume lar

/RMG/Generator/Confinement/Geometrical/AddSolid Cylinder
/RMG/Generator/Confinement/Geometrical/CenterPositionX 0 m
/RMG/Generator/Confinement/Geometrical/CenterPositionY 0 m
/RMG/Generator/Confinement/Geometrical/CenterPositionZ 0.69 m
/RMG/Generator/Confinement/Geometrical/Cylinder/InnerRadius 0 m
/RMG/Generator/Confinement/Geometrical/Cylinder/OuterRadius 0.7 m
/RMG/Generator/Confinement/Geometrical/Cylinder/Height 2.4 m

/RMG/Generator/Select GPS
/gps/particle     opticalphoton
/gps/ene/type     Gauss
/gps/ene/mono     9.68 eV # 128nm (VUV) LAr scintillation
/gps/ene/sigma    0.22 eV # gaussian width
/gps/ang/type     iso
# use random polariztation (this emits warnings that can be ignored)
#/gps/polarization 1 1 1

/run/beamOn 80000000
```

Run remage with this macro to produce the (flat) output file:

```
$ remage -g l200-geometry.gdml -o map.stp.lh5 --flat-output -- map.mac
```

### 2. Convert stp to evt file

Convert the step table output from remage to a more efficiently iterable form:

```
$ reboost-optical evt --detectors detectors.json map.stp.lh5 map.evt.lh5
```

:::{note}

this step might be replaced with GLM iteration in the future.

:::

the `evt` step can be very well parallelized by just running it in parallel for different
files. memory usage for each process should be minimal.

### 3. Create the map

with a map settings file (`map-settings.json`):

```json
{
  "range_in_m": [
    [-0.7, 0.7],
    [-0.7, 0.7],
    [-0.51, 1.89]
  ],
  "bins": [280, 280, 480]
}
```

a list of the optical detector uids has to be provided in a second JSON file (`detectors.json`):

```json
[1052802, 1052803 /* ... */]
```

```
$ reboost-optical createmap --settings map-settings.json --detectors detectors.json map.evt.lh5 map.map.lh5
```

`createmap` can also work on multiple input files at once. Make sure that enough memory is
available; the map object is _fully_ stored in memory. In this example: For the 58
hardware channels, the example above would require

```{math}
\text{memory} = 8 \cdot n_x \cdot n_y \cdot n_z \cdot (n_\text{ch} + 4) = 8 \cdot 280 \cdot 280 \cdot 480 \cdot (58 + 4) = 19\times10^{9} \text{[bytes]}
```

but peak memory usage might be higher. The input buffer will also use some memory.

For parallelization, `createmap` supports a `-N <n_cpu>` argument. However, this cannot be
scaled indefinitely. Be aware of the memory configuration of the compute node used (i.e.,
performance may suffer when using more processors than in a NUMA domain). `createmap`
internally uses locks to access one _shared memory_ instance of the to-be-created map. The
more CPUs are used for the task, the more lock contention might happen.

:::{note}

In practice, a specific scheme has been shown to be working for large LEGEND maps. 16 cores
on a NERSC perlmutter node work quite fine to produce a map from a set of input files. To
use more of the available resources, 256 input files are used for one `createmap` task; up
to 16 (typically only 12 to have some spare memory for the peaks) such tasks are run in
parallel. This means a total of 4096 remage output files can be read in parallel.

The 12-16 output files can then be combined into one with

```
$ reboost-optical mergemap --settings map-settings.json map.map*.lh5 final-output-map.lh5
```

:::

## Applying optical maps to physics simulations

:::{important}

_reboost-optical_ is not yet integrated with the remaining _reboost_ stack. The optical
response can be generated with the standalone command `reboost-optical convolve`.

_reboost-optical_ currently **does not support the recent changes in remage output format**.
It currently only works when the following macro commands are set:

```
/RMG/Output/Scintillator/StoreParticleVelocities
/RMG/Output/Scintillator/StepPositionMode Both
/RMG/Output/Scintillator/Cluster/PreClusterOutputs false
```

:::

If the liquid argon volume is registered as a `Scintillator` detector with uid 1, the
optical response can be created with:

```
reboost-optical convolve --material lar --map map.lh5 --edep remage-output.lh5 --edep-lgdo /stp/det001 --output optical-response.lh5
```
