Blue Brain SNAP
===============

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

There are two main interface classes provided by Blue Brain SNAP:

:class:`.Circuit` corresponds to the *static* structure of a neural network, that is:

- node positions and properties,
- edge positions and properties, and,
- detailed morphologies.

:class:`.Simulation` corresponds to the *dynamic* data for a neural network simulation, including:

- spike reports,
- soma reports, and,
- compartment reports.

Most of Blue Brain SNAP methods return `pandas <https://pandas.pydata.org>`__ Series or DataFrames,
indexed in a way to facilitate combining data from different sources (that is, by node or edge IDs).

Among other dependencies, Blue Brain SNAP relies on Blue Brain Project provided libraries:

- `libsonata <https://github.com/BlueBrain/libsonata>`__, for accessing SONATA files
- `NeuroM <https://github.com/BlueBrain/NeuroM>`__, for accessing detailed morphologies

Tools
-----

Blue Brain SNAP also provides a SONATA circuit validator for verifying circuits.

The validation includes:

- integrity of the circuit config file.
- existence of the different node/edges files and ``components`` directories.
- presence of the "sonata required" field for node/edges files.
- the correctness of the edge to node population/ids bindings.
- existence of the morphology files for the nodes.

This functionality is provided by either the cli function:

.. code-block:: shell

    bluepysnap validate my/circuit/path/circuit_config.json


Or a python free function:

.. code-block:: python3

    from bluepysnap.circuit_validation import validate
    errors = validate("my/circuit/path/circuit_config.json")


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
