HPGe detector simulations
=========================

*reboost-hpge* is a sub-package for post-processing the high purity Germanium detector (HPGe) part of *remage* simulations.
It provides a flexible framework to implement a user customised post-processing.

Command line interface
----------------------

A command line tool *reboost-hpge* is created to run the processing.

.. code-block:: console

    $ reboost-hpge -h

Different modes are implemented to run the tiers. For example to run the *hit* tier processing (more details in the next section).

.. code-block:: console

    $ reboost-hpge hit -h


*remage* lh5 output format
--------------------------

The output simulations from *remage* are described in `remage-docs <https://remage.readthedocs.io/en/stable/output.html>`_.
By default two ``lgdo.Table`` `docs <https://legend-pydataobj.readthedocs.io/en/stable/api/lgdo.types.html#lgdo.types.table.Table>`_ are stored with the
following format:

.. code-block:: console

    /
    └── hit · HDF5 group
        ├── det000 · table{evtid,particle,edep,time,xloc,yloc,zloc}
        │   ├── edep · array<1>{real}
        │   ├── evtid · array<1>{real}
        │   ├── particle · array<1>{real}
        │   ├── time · array<1>{real}
        │   ├── xloc · array<1>{real}
        │   ├── yloc · array<1>{real}
        │   └── zloc · array<1>{real}
        ├── det001 · table{evtid,particle,edep,time,xloc,yloc,zloc}
        |    ....
        |    ....
        └── vertices · table{evtid,time,xloc,yloc,zloc,n_part}
            ├── evtid · array<1>{real}
            ├── n_part · array<1>{real}
            ├── time · array<1>{real}
            ├── xloc · array<1>{real}
            ├── yloc · array<1>{real}
            └── zloc · array<1>{real}



One table is stored per sensitive Germanium detector and a Table of the vertices is also stored.
All the data is stored as (flat) 1D arrays.

- *edep*:  energy deposited in Germanium (in keV).
- *evtid*: index of the simulated event,
- *particle*: Geant4 code for the particle type,
- *time*: time of the event relative to the start of the event,
- *xloc/yloc/xzloc*: Position of the interaction / vertex,
- *n_part*: Number of particles emitted.

However, this format is not directly comparable to experimental data.


Data tiers
----------

The processing is defined in terms of several *tiers*:

- **stp** or "step" the raw *remage* outputs corresponding to Geant4 steps,
- **hit** the data from each channel independently after grouping in discrete physical interactions in the detector.
- **evt** or "event" the data combining the information from various detectors.

The processing is divided into two steps :func:`build_hit`  ``build_evt`` [WIP].

Hit tier processing
-------------------

The processing is based on a YAML or JSON configuration file. For example:

.. code-block:: json

               {
                    "channels": [
                        "det000",
                        "det001",
                        "det002",
                        "det003"
                    ],
                    "outputs": [
                        "t0",
                        "truth_energy_sum",
                        "smeared_energy_sum",
                        "evtid"
                    ],
                    "step_group": {
                        "description": "group steps by time and evtid.",
                        "expression": "reboost.hpge.processors.group_by_time(stp,window=10)"
                    },
                    "locals": {
                        "hpge": "reboost.hpge.utils(meta_path=meta,pars=pars,detector=detector)"
                    },
                    "operations": {
                        "t0": {
                            "description": "first time in the hit.",
                            "mode": "eval",
                            "expression": "ak.fill_none(ak.firsts(hit.time,axis=-1),np.nan)"
                        },
                        "truth_energy_sum": {
                            "description": "truth summed energy in the hit.",
                            "mode": "eval",
                            "expression": "ak.sum(hit.edep,axis=-1)"
                        },
                        "smeared_energy_sum": {
                            "description": "summed energy after convolution with energy response.",
                            "mode": "function",
                            "expression": "reboost.hpge.processors.smear_energies(hit.truth_energy_sum,reso=pars.reso)"
                        }

                    }
                }

It is necessary to provide several sub-dictionaries:

- **channels**: list of HPGe channels to process.
- **outputs**: list of fields for the output file.
- **locals**: get objects used by the processors (passed as ``locals`` to ``LGDO.Table.eval``)
- **step_group**: this should describe the function that groups the Geant4 steps into physical *hits*.
- **operations**: further computations / manipulations to apply.

The **step_group** block sets the structure of the output file, this function reformats the flat input table into a table
with a jagged structure where each row corresponds to a physical hit in the detector. For example:

.. code-block:: console

    evtid: [0    ,     0,     1, ... ]
    edep:  [101.2, 201.2, 303.7, ... ]
    time:  [0    , 0.1  , 0,     ... ]
    ....

Becomes a Table of ``VectorOfVectors`` with a jagged structure. For example:

.. code-block:: console

    evtid: [[0    ,     0], [    1],[...],... ]
    edep:  [[101.2, 201.2], [303.7],[...],... ]
    time:  [[0    ,   0.1], [    0],[...],... ]
    ....

The recommended tool to manipulate jagged arrays is awkward `[docs] <https://awkward-array.org/doc/main/>`_ and much of *reboost* is based on this.


It is necessary to chose a function to perform this step grouping, this function must take in the *remage* output table and return
a table where all the input arrays are converted to ``LGDO.VectorOfVectors`` with a jagged structure. In the expression of the function *stp* is an alias
for the input *remage* Table. This then must return the original LH5 table with the same fields as above restructured so each field is a ``VectorOfVectors``.
In addition a ``global_evtid`` field is adding which represents the index of the event over all input files.

Next a set of operations can be specified, these can perform any operation that doesn't change the length of the data. They can be either basic numerical operations
(including awkward or numpy) or be specified by a function. The functions can reference several variables:

- **hit** the output table of step grouping (note that the table is constantly updated so the order of operations is important),
- **pars** a named tuple of parameters (more details later) for this detector,
- **hpge** the ``legendhpges.HPGe`` object for this detector,
- **phy_vol** the ``pygometry`` physical volume for the detector.

Finally the outputs field specifies the columns of the Table to include in the output table.

lh5 i/o operations
------------------

:func:`build_hit` contains several options to handle i/o of lh5 files.

Typically raw geant4 output files can be very large (many GB) so it is not desirable or feasible to read the full file into memory.
Instead the :class:`lgdo.lh5.LH5Ierator` is used to handle iteration over chunks of files keeping memory use reasonable. The *buffer* keyword argument
to :func:`build_hit` controls the size of the buffer.

It is possible to specify a list of files of use wildcards, the *merge_input_files* argument controls whether the outputs are merged or kept as separate files.

Finally, it is sometimes desirable to process a subset of the simulated events, for example to split the simulation by run or period. The *n_evtid* and *start_evtid*
keywords arguments control the first simulation index to process and the number of events. Note that the indices refer to the *global* evtid when multiple files are used.

parameters and other *local* variables
--------------------------------------

Often it is necessary to include processors that depend on parameters (which) may vary by detector. To enable this the user can specify a dictionary of
parameters with the *pars* keyword, this should contain a sub-dictionary per detector for example:

.. code-block:: json

                {
                    "det000": {
                        "reso": 1,
                        "fccd": 0.1,
                        "phy_vol_name":"det_phy",
                        "meta_name": "icpc.json"
                    }
                }

This dictionary is internally converted into a python ``NamedTuple`` to make cleaner syntax. The named tuple for each detector is then passed as a
``local`` dictionary to the evaluation of the operations with name "pars".

In addition, for many post-processing applications it is necessary for the processor functions to know the geometry. This is made possible
by passing the path to the GDML file and the path to the metadata ("diodes" folder) with the *gdml* and *meta_path* arguments to build_hit.
From the GDML file the ``pyg4ometry.geant4.Registry`` is extracted.

To allow the flexibility to write processors depending on arbitrary (more complicated python objects), it is possible to add the *locals* dictionary
to the config file. The code will then evaluate the supplied expression for each sub-dictionary. These expressions can depend on:

- the *remage* detector name: "detector",
- the path to the metadata: "meta",
- the geant4 registry: "reg",
- the parameters for this detector: "pars".

These expressions are then evaluated (once per detector) and added to the *locals* dictionary of ``Table.eval``, so can be references in the expressions.

For example one useful object for post-processing is the `legendhpges.base.HPGe <https://legend-pygeom-hpges.readthedocs.io/en/latest/api/legendhpges.html#legendhpges.base.HPGe>`_ object for the detector.
This can be constructed from the metadata using.

.. code-block:: json

    {"hpge": "reboost.hpge.utils(meta_path=meta,pars=pars,detector=detector)"}

This will then create the hpge object for each detector and add it to the "locals" mapping of "eval" so it can be used.

Possible intended use case of this functionality are:

 - extracting detector mappings (eg drift time maps),
 - extracting the kernel of a machine learning model.
 - any more complicated (non-JSON serialisable objects).

adding new processors
---------------------

Any python function can be a ``reboost.hit`` processor. The only requirement is that it should return a :class:`VectorOfVectors`, :class:`Array`` or :class:`ArrayOfEqualSizedArrays`
with the same length as the hit table. This means processors can act on subarrays (``axis=-1`` in awkward syntax) but should not combine multiple rows of the hit table.
