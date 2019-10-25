Blue Brain SNAP
==============

Blue Brain Simulation and Neural network Analysis Productivity layer (Blue Brain SNAP).

Blue Brain SNAP is a Python library for accessing BlueBrain circuit models represented in
`SONATA <https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md>`__ format.

|build_status| |coverage|

Installation
------------

Blue Brain SNAP can be installed using ``pip``::

   pip install bluepysnap

Usage
-----

The main interface class exposed is ``Circuit``, which corresponds to the *static* structure of a neural network, that is:

- node positions and properties
- edge positions and properties
- detailed morphologies

Most of Blue Brain SNAP methods return `pandas <https://pandas.pydata.org>`__ Series or DataFrames,
indexed in a way to facilitate combining data from different sources (that is, by node or edge IDs).

Among other dependencies, Blue Brain SNAP relies on Blue Brain Project provided libraries:

- `libsonata <https://github.com/BlueBrain/libsonata>`__, for accessing SONATA files
- `NeuroM <https://github.com/BlueBrain/NeuroM>`__, for accessing detailed morphologies

License
-------

Blue Brain SNAP is licensed under the LGPLv3. Refer to the
`LICENSE.txt <https://github.com/BlueBrain/snap/blob/master/LICENSE.txt>`__ for details.

.. |build_status| image:: https://travis-ci.com/BlueBrain/snap.svg?branch=master
   :target: https://travis-ci.com/BlueBrain/snap
   :alt: Build Status

.. |coverage| image:: https://codecov.io/github/BlueBrain/snap/coverage.svg?branch=master
   :target: https://codecov.io/github/BlueBrain/snap?branch=master
   :alt: codecov.io
