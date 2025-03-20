
.. _processors:

Using reboost processors
========================

A *processor* in *reboost* is a function computing a new quantity of interest. For example,
the energy deposited after correction for the HPGe surface responses.

A special set of processors handles grouping of the "steps" from remage, i.e. discrete energy depositions
into the detectors into "hits". A hit corresponds to a physical interaction in the detector, related to the
time-resolution of the detector.

Step-grouping
-------------

A special class of processors implements the conversion of steps into the basic hit tier output.
These processors convert the "regular" structure of the remage output into a jagged structure, where each row corresponds
to a "hit" and the fields are variable length vectors containing information on the steps in each hit.

Currently two step-grouping options are implemented. Once the remage output is loaded as an *awkward* array, we can either
group by Geant4 event id or by time and event id.

.. code-block:: python

    # read the data with awkward
    data = lh5.read(f"stp/{det_name}", file_path).view_as("ak")

    # group by evtid
    hits_by_evtid = group_by_evtid(data)

    # group also by time
    hits_by_time = group_by_time(data, window=10)  # unit is us


Other processors
----------------

Additional *reboost* processors compute further quantities of interesting, this can consist of:

- reduction (e.g. summing over steps),

The only prescriptiion for a *reboost* processor is that the function should return either an :class:`lgdo.LGDO` object,
or an :class:`awkward.Array`. These processors should not change the length of the object, i.e. they should only act on axes
more than 1. Documentation describing the various processors is contained in the API documentation.
