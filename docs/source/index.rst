Welcome to reboost's documentation!
==========================================

*reboost* is a python package for the post-processing of `remage <https://remage.readthedocs.io/en/stable/>`_ monte-carlo Simulations.

Getting started
---------------

*reboost* can be installed with *pip*:

.. code-block:: console

   $ git clone git@github.com:legend-exp/reboost.git
   $ cd reboost
   $ pip install .

*reboost* is currently divided into two programs:
 - *reboost-optical* for processing optical simulations,
 - *reboost-hpge* for processing HPGe detector simulations.

Both can be run on the command line with:

.. code-block:: console

   $ reboost-optical -h
   $ reboost-hpge -h

Next steps
----------

.. toctree::
   :maxdepth: 2

   User Manual <manual/index>

.. toctree::
   :maxdepth: 1

   Package API reference <api/modules>


See also
--------
 - `remage <https://remage.readthedocs.io/en/stable/>`_: Modern *Geant4* application for HPGe and LAr experiments,
 - `legend-pygeom-hpges <https://legend-pygeom-hpges.readthedocs.io/en/latest/>`_: Package for handling HPGe detector geometry in python,
 - `pyg4ometry <https://pyg4ometry.readthedocs.io/en/stable/>`_: Package to create simulation geometry in python,
 - `legend-pygeom-optics <https://legend-pygeom-optics.readthedocs.io/en/stable/>`_: Package to handle optical properties in python,
 - `legend-pygeom-l200 <https://github.com/legend-exp/legend-pygeom-l200>`_: Implementation of the LEGEND-200 experiment (**private**),
 - `pyvertexgen <https://github.com/tdixon97/pyvertexgen/>`_: Generation of vertices for simulations.
