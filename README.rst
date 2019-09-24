BlueBrain SNAP
==============

BlueBrain Simulation and Neural network Analysis Productivity layer (BlueBrain SNAP).

BlueBrain SNAP is a Python library for accessing BlueBrain circuit models represented in
`SONATA <https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md>`__ format.

|build_status| |coverage|

Installation
------------

BlueBrain SNAP can be installed using ``pip``::

   pip install bluepysnap

Usage
-----

The main interface class exposed is ``Circuit``, which corresponds to the *static* structure of a neural network, that is:

- node positions and properties
- edge positions and properties
- detailed morphologies

Most of BlueBrain SNAP methods return `pandas <https://pandas.pydata.org>`__ Series or DataFrames,
indexed in a way to facilitate combining data from different sources (that is, by node or edge IDs).

Among other dependencies, BlueBrain SNAP relies on BlueBrain Project provided libraries:

- `libsonata <https://github.com/BlueBrain/libsonata>`__, for accessing SONATA files
- `NeuroM <https://github.com/BlueBrain/NeuroM>`__, for accessing detailed morphologies

License
-------

BlueBrain SNAP is licensed under the LGPL. Refer to the
`LICENSE.txt <https://github.com/BlueBrain/snap/blob/master/LICENSE.txt>`__ for details.

.. |build_status| image:: https://travis-ci.com/BlueBrain/snap.svg?branch=master
   :target: https://travis-ci.com/BlueBrain/snap
   :alt: Build Status

.. |coverage| image:: https://codecov.io/github/BlueBrain/snap/coverage.svg?branch=master
   :target: https://codecov.io/github/BlueBrain/snap?branch=master
   :alt: codecov.io
