(config)=

# The hit tier configuration file

_reboost_ can also be run with a YAML or JSON based configuration file to manage
the full post-processing in a similar way to _pygama_. The configuration file
format is chosen to be very generic to allow an arbitrary post-processing.

This config file has several necessary sections:

```yaml
objects:
  geometry: pyg4ometry.gdml.Reader(ARGS.gdml).getRegistry()
  user_pars: dbetto.AttrsDict(dbetto.utils.load_dict(ARGS.pars))
```

This creates any objects needed for the post-processing. Each value can be a
generic python expression producing any python object. For example here we
extract the {class}`pyg4ometry.geant4.registry` for the geometry, and a
dictionary of user parameters.

The ARGS keyword refers to a dictionary of user supplied arguments (more details
later.)

Next it is necessary to define a series of processing groups, for example to
allow a different post-processing to Germanium detectors and SiPMs. For each
processing group we should define a detector mapping.

In most cases this is effectively just a list of remage input tables, for
example:

```yaml
processing_groups:
    - name: geds

    detector_mapping:
        - output: det001
        - output: det002
```

Since if no "input" key is supplied to the dictionary it is assumed the input
and output detector have the same name. It is also possibly to supply an
expression to be evaluated, for example using _legend-metadata_.

Finally, for some type of detector the input and the output detectors do not
have the same name, for example for LAr simulations, the input table is the LAr
volume and the output is a SiPM. This can be handled by specifying the input
table name. For example:

```yaml
processing_groups:
    - name: sipms

    detector_mapping:
        - output: det003
          input: LAr
```

Next for every detector we have the option to define (generically) a set of
objects useful for the post-processing. Again the values in the dictionary are
generic python expressions depending on:

- _DETECTOR_: The name of the detector,
- _OBJECTS_: The previously defined dictionary of user supplied objects.

Below is an example of the type of objects you might extract. This generic
syntax allows the user to include any arbitrary input parameters or objects.

```yaml
detector_objects:
  meta: dbetto.AttrsDict(pygeomtools.get_sensvol_metadata(OBJECTS.geometry,
    OBJECTS.user_pars[DETECTOR].name ))
  pyobj:
    legendhpges.make_hpge(pygeomtools.get_sensvol_metadata(OBJECTS.geometry,OBJECTS.user_pars[DETECTOR].name),
    registry = OBJECTS.reg)
  phyvol: OBJECTS.geometry.physicalVolumeDict[OBJECTS.user_pars[DETECTOR].name]
  det_pars: OBJECTS.user_pars[DETECTOR]
```

We then give a list of fields to include in the output file.

```yaml
outputs:
  - t0
  - evtid
  - truth_energy
  - active_energy
  - smeared_energy
```

Next we define the step grouping function (i.e. we create our hits). Here
"STEPS" is a special keyword corresponding to the remage data.

```yaml
hit_table_layout: reboost.shape.group.group_by_evtid(STEPS)
```

Finally, we provide a list of processors (operations to be evaluated). These can
depend on:

- _HITS_: the table of hits after step grouping,
- _OBJECTS/GLOBAL_OBJECTS_: The user defined objects from earlier.

The expressions are applied sequentially, in each case adding a new row to the
HITS table. An example simple post-processing chain is shown below:

```yaml
operations:
  t0: ak.fill_none(ak.firsts(HITS.time, axis=-1), np.nan)
  first_evtid: ak.fill_none(ak.firsts(HITS.evtid, axis=-1), np.nan)
  truth_energy: ak.sum(HITS.edep, axis=-1)

  distance_to_nplus:
    reboost.hpge.surface.distance_to_surface( HITS.xloc, HITS.yloc, HITS.zloc,
    DETECTOR_OBJECTS.pyobj, DETECTOR_OBJECTS.phyvol.position.eval(),
    distances_precompute = HITS.dist_to_surf, precompute_cutoff = 2/1000,
    surface_type='nplus')

  activeness:
    reboost.math.functions.piecewise_linear_activeness(HITS.distance_to_nplus,
    fccd=DETECTOR_OBJECTS.det_pars.fccd_in_mm,
    tl=DETECTOR_OBJECTS.det_pars.tl_in_mm)

  active_energy: ak.sum(HITS.edep*HITS.activeness, axis=-1)
  smeared_energy: reboost.math.stats.gaussian_sample(HITS.active_energy,DETECTOR_OBJECTS.det_pars.reso_in_sigma)
```

The post-processing based on this configuration file format is handled by
{func}`reboost.build_hit.build_hit`, we also wrote a command line interface
which can be run with:

```console
reboost build-hit -h
```
