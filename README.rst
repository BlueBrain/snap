BlueBrain `SNAP` is a Python library for accessing `BlueBrain <https://github.com/bluebrain/>`__ circuit models represented in `SONATA <https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md>`__ format.

|build_status| |coverage|


The main interface class exposed is `Circuit`, which corresponds to the *static* structure of a neural network, i.e.:
 - node positions / properties
 - edge positions / properties
 - detailed morphologies

Most of `SNAP` methods return `pandas <https://pandas.pydata.org>`__ Series or DataFrames, indexed in a way to facilitate combining data from different sources (i.e. by node or edge IDs).

Among other dependencies, `SNAP` relies on BBP-provided libraries:
 - `libsonata <https://github.com/BlueBrain/libsonata>`__, for accessing SONATA files
 - `NeuroM <https://github.com/BlueBrain/NeuroM>`__, for accessing detailed morphologies


.. |build_status| image:: https://travis-ci.com/BlueBrain/snap.svg?branch=master
   :target: https://travis-ci.com/BlueBrain/snap
   :alt: Build Status

.. |coverage| image:: https://codecov.io/github/BlueBrain/snap/coverage.svg?branch=master
   :target: https://codecov.io/github/BlueBrain/snap?branch=master
   :alt: codecov.io
