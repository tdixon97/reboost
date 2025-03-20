(iteration)=

# Efficient iteration over remage files

:::{note}
This step is only relevant for applications involving large simulation
files, or where the reboost configuration file option is used.
:::

Simulation files are often very large. This poses a challenge in processing the
data, since it may not always be feasible to read the full simulated file into
memory.

In addition, not every simulated event will lead to energy deposits in the
detectors, and each simulated event will have multiple rows in the output file
(corresponding to the different) steps. This introduces a challenge in iterating
over the files, since a naive iteration could cause a simulated event to be
split between two chunks.

Finally, it is often desirable to split the initial remage files (for example to
split the file by run), to do this it is necessary to be able to read the data
only corresponding to a range of simulated event ids.

The _geant4 event lookup map_ (or **glm**) is a solution to these issues. The
**glm** is a {func}`LGDO.VectorOfVectors` with three fields:

- _g4_evtid_: The index of the simulated event,
- _n_rows_: The number of rows of the output file corresponding to this event,
- _start_row_: The first row of the file.

:::{note}

- This object should be computed once per output table (detector).
- Geant4 events that do not lead to rows in the output file are still included.
  :::

The **glm** can be build from the remage outputs using the
{func}`reboost.build_glm.build_glm` function. A command line interface has also
been written:

```console
reboost build-glm -h
```

We have created a {class}`reboost.iterator.GLMIterator` class to allow for easy
iteration over the stp files using the glm.
