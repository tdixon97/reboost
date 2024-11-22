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

The processing is defined in terms of several *tiers*, mirroring the logic of the `pygama <https://pygama.readthedocs.io/en/stable/>`_ data processing software used for LEGEND.

- **stp** or "step" the raw *remage* outputs corresponding to Geant4 steps,
- **hit** the data from each channel independently after grouping in discrete physical interactions in the detector.
- **evt** or "event" the data combining the information from various detectors (includes generating the **tcm** or time-coincidence map).

The processing is divided into two steps :func:`build_hit`  ``build_evt`` [WIP].

Hit tier processing
-------------------

The hit tier converts the raw remage file based on Geant4 steps to a file corresponding to the physical interactions in the detectors.
Only steps corresponding to individual detectors are performed in this step.
The processing is based on a YAML or JSON configuration file. For example:

.. code-block:: yaml

    channels:
     - det000
     - det001
     - det002
     - det003

    outputs:
     - t0
     - truth_energy_sum
     - smeared_energy_sum
     - evtid

    step_group:
        description: group steps by time and evtid.
        expression: 'reboost.hpge.processors.group_by_time(stp,window=10)'
    locals:
        hpge: 'reboost.hpge.utils(meta_path=meta,pars=pars,detector=detector)'

    operations:
        t0:
            description: first time in the hit.
            mode: eval
            expression: 'ak.fill_none(ak.firsts(hit.time,axis=-1),np.nan)'
        truth_energy_sum:
            description: truth summed energy in the hit.
            mode: eval
            expression: 'ak.sum(hit.edep,axis=-1)'
        smeared_energy_sum:
            description: summed energy after convolution with energy response.
            mode: function
            expression: |
            reboost.hpge.processors.smear_energies(hit.truth_energy_sum,reso=pars.reso)

It is necessary to provide several sub-dictionaries:

- **channels**: list of HPGe channels to process.
- **outputs**: list of fields for the output file.
- **locals**: get objects used by the processors (passed as ``locals`` to ``LGDO.Table.eval``), more details below.
- **step_group**: this should describe the function that groups the Geant4 steps into physical *hits*.
- **operations**: further computations / manipulations to apply.

The **step_group** block sets the structure of the output file, this function reformats the flat input table into a table
with a jagged structure where each row corresponds to a physical hit in the detector. For example:

.. code-block:: console

    evtid: [0    ,     0,     1, ... ]
    edep:  [101.2, 201.2, 303.7, ... ]
    time:  [0    , 0.1  , 0,     ... ]
    ....

Becomes a :class:`Table`` of :class:`VectorOfVectors` with a jagged structure. For example:

.. code-block:: console

    evtid: [[0    ,     0], [    1],[...],... ]
    edep:  [[101.2, 201.2], [303.7],[...],... ]
    time:  [[0    ,   0.1], [    0],[...],... ]
    ....

The recommended tool to manipulate jagged arrays is awkward `[docs] <https://awkward-array.org/doc/main/>`_ and much of *reboost* is based on this.


It is necessary to chose a function to perform this step grouping, this function must take in the *remage* output table and return
a table where all the input arrays are converted to :class:`LGDO.VectorOfVectors` with a jagged structure. In the expression of the function *stp* is an alias
for the input *remage* Table. This then must return the original LH5 table with the same fields as above restructured so each field is a :class:`VectorOfVectors`.
In addition a ``global_evtid`` field is adding which represents the index of the event over all input files.

Next a set of operations can be specified, these can perform any operation that doesn't change the length of the data. They can be either basic numerical operations
(including awkward or numpy) or be specified by a function. The functions can reference several variables:

- **hit** the output table of step grouping (note that the table is constantly updated so the order of operations is important),
- **pars** a named tuple of parameters (more details later) for this detector,
- **hpge** the ``legendhpges.HPGe`` object for this detector,
- **phy_vol** the ``pygometry`` physical volume for the detector.

Finally the outputs field specifies the columns of the Table to include in the output table.

lh5 i/o operations
^^^^^^^^^^^^^^^^^^

:func:`build_hit` contains several options to handle i/o of lh5 files.

Typically raw geant4 output files can be very large (many GB) so it is not desirable or feasible to read the full file into memory.
Instead the :class:`lgdo.lh5.LH5Ierator` is used to handle iteration over chunks of files keeping memory use reasonable. The *buffer* keyword argument
to :func:`build_hit` controls the size of the buffer.

It is possible to specify a list of files of use wildcards, the *merge_input_files* argument controls whether the outputs are merged or kept as separate files.

Finally, it is sometimes desirable to process a subset of the simulated events, for example to split the simulation by run or period. The *n_evtid* and *start_evtid*
keywords arguments control the first simulation index to process and the number of events. Note that the indices refer to the *global* evtid when multiple files are used.

parameters and other *local* variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Often it is necessary to include processors that depend on parameters (which) may vary by detector. To enable this the user can specify a dictionary of
parameters with the *pars* keyword, this should contain a sub-dictionary per detector for example:

.. code-block:: yaml

    det000:
     reso: 1
     fccd: 0.1
     phy_vol_name: det_phy
     meta_name: icpc.json


This dictionary is internally converted into a python ``NamedTuple`` to make cleaner syntax. The named tuple for each detector is then passed as a
``local`` dictionary to the evaluation of the operations with name "pars".

In addition, for many post-processing applications it is necessary for the processor functions to know the geometry. This is made possible
by passing the path to the GDML file and the path to the metadata ("diodes" folder) with the *gdml* and *meta_path* arguments to build_hit.
From the GDML file the ``pyg4ometry.geant4.Registry`` is extracted.

To allow the flexibility to write processors depending on arbitrary (more complicated python objects), it is possible to add the *locals* dictionary
to the config file. The code will then evaluate the supplied expression for each sub-dictionary. These expressions can depend on:

- **detector**: the *remage* detector name,
- **meta**: the path to the metadata,
- **reg**: the geant4 registry,
- **pars**: the parameters for this detector.

These expressions are then evaluated (once per detector) and added to the *locals* dictionary of :func:`Table.eval`, so can be references in the expressions.

For example one useful object for post-processing is the :class:`legendhpges.base.HPGe` object for the detector.
This can be constructed from the metadata using.

.. code-block:: yaml

    hpge: 'reboost.hpge.utils(meta_path=meta,pars=pars,detector=detector)'

This will then create the hpge object for each detector and add it to the "locals" mapping of "eval" so it can be used.

Possible intended use case of this functionality are:

 - extracting detector mappings (eg drift time maps),
 - extracting the kernel of a machine learning model.
 - any more complicated (non-JSON serialisable objects).

Adding new processors
^^^^^^^^^^^^^^^^^^^^^

Any python function can be a ``reboost.hit`` processor. The only requirement is that it should return a:

- :class:`VectorOfVectors`,
- :class:`Array` or
- :class:`ArrayOfEqualSizedArrays`

with the same length as the hit table. This means processors can act on subarrays (``axis=-1`` in awkward syntax) but should not combine multiple rows of the hit table.

It is simple to accommodate most of the current and future envisiged post-processing in this framework. For example:

- clustering hits would result in a new :class:`VectorOfVectors` with the same number of rows but fewer entries per vector,
- pulse shape simulations to produce waveforms (or ML emmulation of this) would give an :class:`ArrayOfEqualSizedArrays`,
- processing in parallel many parameters (eg. for systematic) studies would give a nested :class:`VectorOfVectors`.

Time coincidence map (TCM)
--------------------------

The next step in the processing chain is the **event** tier, this combines the information from the various sub-systems to produce detector wide events.
However, before we can generate the *evt* tier we need to generate the "time-coincidence-map". This determines which of the hits in the various detectors
are occurring *simultaneously* (actually within some coincidence time window) and should be part of the same event.
Some information on the TCM in data is given in `[pygama-evt-docs] <https://pygama.readthedocs.io/en/stable/api/pygama.evt.html#>`_. The *reboost* TCM is fairly similar.

The generation of the TCM is performed by :func:`reboost.hpge.tcm.build_tcm` which generates and stores the TCM on disk.

.. warning::
    The generation of the TCM from the times of hits is slightly different to the "hardware-tcm" used for LEGEND physics data. In the experimental data, a signal on one channel, triggers
    the readout of the full array. Care should be taken for deecays or interactions with ~ :math:`10-100 \mu s` time differences between hits.
    However, in practice for most cases the time differences are very small and the two TCM should be equivalent after removing hits below threshold.

Before explaining how the TCM is constructed we make a detour to explain the different indices present in the reboost and remage files.

- **stp.evtid**: in the remage output files we have a variable called evtid. This is the index of the decay, so as explained earlier a single evtid can result in multiple hits in the detector.
- **hit.global_evtid**: However, when multiple files are read the evtid are now no longer necessarily sorted or unique and so we define a new index in the hit tier. This is extracted from the vertices table as
    the sum of the number of evtid in the previous files plus the row in the vertex table of the evtid. A vector of vectors called "hit._global_evtid" is added to the hit table. We can also extract
    a flat array of indices (for easier use in the evt tier) with a simple processor:

    .. code:: yaml

        global_evtid:
            description: global evtid of the hit.
            mode: eval
            expression: 'ak.fill_none(hit._global_evtid,axis=-1),np.nan)'


    This field is mandatory to generate the TCM, and the name of the field is an argument to "build_tcm".
- **hit idx**: Multiple rows in the hit table may contain the same "global_evtid" while many "global_evtid" do not result in a hit. The hit idx is just the row of the hit table a hit corresponds to.
- **channel_id**: When we combine multiple channels we assign them an index, this is set in the original remage macro file.

:func:`build_tcm` saves two VectorOfVectors (with the same shape) to the output file, corresponding to the **channel_id** and the **hit_idx** of each event.

.. note::
    - This storage is slightly different to the TCM in data, but is chosen to allow easy iteration through the TCM.
    - We do not currently support merging multiple **hit** tier files, this is since then the TCM would need to know which file each hit corresponded to.

Event tier processing
---------------------

The event tier combines the information from various detector systems. Including in future the optical detector channels. This step is thus only necessary for experiments with
many output channels.

The processing is again based on a YAML or JSON configuration file. Most of the work to evaluate each expression is done by the :func:`pygama.evt.build_evt.evaluate_expression` and our conventions for processors
follow those for pygama.
The input configuration file is identical to a pygama evt tier configuration file (see an example in :func:`pygama.evt.build_evt.build_evt`).

For example:

.. code-block:: yaml

        channels:
            geds_on:
                - det000
                - det001
                - det002
            geds_ac:
                - det003

        outputs:
         - energy
         - multiplicity

        operations:
            energy_id:
                channels: geds_on
                aggregation_mode: gather
                query: hit.energy > 25
                expression: tcm.channel_id
            energy:
                aggregation_mode: 'keep_at_ch:evt.energy_id'
                channels: geds_on
                expression: hit.energy
            multiplicity:
                channels:
                    - geds_on
                    - geds_ac
                aggregation_mode: sum
                expression: hit.energy > 25
                initial: 0

- **channels**  : defines a set of groups of channel names which the operations will be applied to.
- **outputs**   : defines the fields to include in the output file.
- **operations**: a list of operations to perform.

The type of operations is based on the "evaluation modes of" :func:`pygama.evt.build_evt.build_evt`.
Each operation is defined by a configuration block which can have the following keys:

- **channels**: list of channels to perform the operation on,
- **exlude_channels**: channels to set to the default value,
- **initial**: initial value of the aggregator,
- **aggregation_mode**: how to combine the channels (more information below),
- **expression**: expression to evaluate,
- **query**: logical statement to only select some channels,
- **sort**: expression used for sorting the output, format of "ascend_by:field" or "descend_by:field".



Aggregation modes
^^^^^^^^^^^^^^^^^

There are several different ways to aggregate the data from different detectors / channels.

- *"no aggregator supplied"* : then the code will perform a simple evaluation of quantities in the ``evt`` tier data for example:

    .. code-block:: yaml

        energy_sum:
            expression: ak.sum(evt.energies,axis=-1)

- *"first_at:sorter"* picks the value corresponding to the channel (TCM ID) with the lowest value of the "sorter" field. For example:

    .. code-block:: yaml

        first_time:
            channels: geds_on
            aggregation_mode: first_at:hit.timestamp
            expression: hit.timestamp

- *"last_at:sorter"* similar for the highest value,
- *"gather"*: combines the fields into a :class:`VectorOfVectors`, sorted by the "sort" keys. For example the the following processor is used to extract the channel id (`tcm.array_id`) for every hit above a 25 keV energy threshold.

    .. code-block:: yaml

        channel_id:
            channels: geds_on
            aggregation_mode: gather
            query: hit.energy > 25
            expression: tcm.array_id
            sort: descend_by:hit.energy

- *"keep_at_channel:channel_id_field"*: similarly combines into a :class:`VectorOfVectors`, however uses only the ids from the "channel_id_field" and preserves the order of the subvectors.
    For example we can make a processor to extract the energy of each hit from the previous part.

    .. code-block:: yaml

        energy:
            channels: geds_on
            aggregation_mode: keep_at_channel:evt.array_id
            expression: hit.energy

- *"keep_at_idx:tcm_index_field"* similar but instead preserves the shape of the tcm, first we need to generate a tcm index field.

     .. code-block:: yaml

        tcm_idx:
            channels: geds_on
            aggregation_mode: gather
            query: hit.energy > 25
            expression: tcm.index
            sort: descend_by:hit.energy

    this says for every element in the :class:`VectorOfVectors` which index in the flattened data of the tcm to extract the value from. So to find the value to fill a row in the output the code will search for the
    tcm `idx`` and `id` corresponding to the supplied index.

- *"all"*, *"any"*, *"sum"* aggregates by the operation.

There is also a function mode, but this is not currently used in reboost, and is not expected to be needed.
