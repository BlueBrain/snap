Changelog
=========

Version v0.13.0
---------------

New Features
~~~~~~~~~~~~~
- Sonata BBP spec:

  - Node/edge populations are now supported in config
  - Population type available in NodePopulation/EdgePopulation
  - Population config (if given) overwrites the "components" config for that population
  - Alternate morphology directories (.h5, .asc) are now supported

Improvements
~~~~~~~~~~~~~~
- Update circuit validation for the current BBP sonata spec

Bug Fixes
~~~~~~~~~
- Fix circuit validation. Validation of morphologies was skipped when no rotations fields were
  present.


Version v0.12.1
---------------

New Features
~~~~~~~~~~~~~
- Adding the h5 and csv file accessors to the Node/EdgeStorage classes.

Bug Fixes
~~~~~~~~~
- Fix the morphology/model access using a numpy int (using a numpy integer to access
  the morphology/model used to fail).

Others
~~~~~~
- Update the copyright.


Version v0.12.0
---------------

Improvements
~~~~~~~~~~~~~~
- removing the MORPH_CACHE_SIZE
- removing neurom as the main reader for morphologies
- adding morphio as the main reader for the morphologies


Version v0.11.0
---------------

New Features
~~~~~~~~~~~~~
- Implement queries mechanism for edges

Improvements
~~~~~~~~~~~~~~
- Pinned major versions of dependencies.

Bug Fixes
~~~~~~~~~
- Pinned major versions of neuroM to <2.0.0.


Version v0.10.0
---------------

New Features
~~~~~~~~~~~~~
- Added NeuronModelsHelper to access nodes neuron models

Improvements
~~~~~~~~~~~~~~
- Moved nodes query mechanism to a separate module

Version v0.9.1
--------------

Bug Fixes
~~~~~~~~~
- Ensure the dtypes as int64 for the node/edge ids (#121).


Version v0.9.0
---------------

New Features
~~~~~~~~~~~~~
- Added a Edges interface to query edges regardless of the population names (#112)
- Added a CircuitEdgeIds object to contain the edge circuit ids (#112)
- Added a ids function to the EdgePopulation class to keep the Edge/Node class homogeneous (#112, #115)
- Added a get function to replace the properties function to the EdgePopulation class to keep the Edge/Node class homogeneous (#113)
- Added a network.py module with a NetworkObject abstract class to factorize the Nodes and Edges classes (#113, #114)
- Added a _doctool.py module with a DocSubstitutionMeta class to update inherited class docstrings (#113)

Deprecation
~~~~~~~~~~~~
- Deprecated the properties function from the EdgePopulation (#113)


Version v0.8.0
---------------

Improvements
~~~~~~~~~~~~~~
- Added the python3.8 toxenv

Removed
~~~~~~~~
- Dropped Python2 support (#109)
  - Removed python2 tox
  - Removed python2 dependencies and bump deps version
  - Removed the python2 switches in setup.py
- Removed all deprecated functions
- Removed six dependency (#110)

Bug Fixes
~~~~~~~~~
- Fixed circuit validation for h5py>=3.0.0


Version v0.7.1
---------------

New Features
~~~~~~~~~~~~~
- Allowed usage of config dict instead of file only (#108)


Version v0.7.0
---------------

New Features
~~~~~~~~~~~~~
- Added a circuit node interface (#99)
  - Added the CircuitNodeId/CircuitNodeIds
  - Added Nodes class
- All functions can use the CircuitNodeId/CircuitNodeIds


Version v0.6.2
---------------

Improvements:
~~~~~~~~~~~~~~
- Update of the example notebooks (#88)
- Improved _check_ids performance (#92)
- Added information about the python3.7 support (#93)
- Moved the CI from travis to GH Actions (#100, #101, #102, #103)

Bug Fixes
~~~~~~~~~
- Fixed unit tests on Mac, fix doc indentation (#91)
- Fixed validation of required datasets of virtual node groups (#98)
- Fixed h5py dependency to be less than 3.0 (#98)


Version v0.6.1
---------------

New Features
~~~~~~~~~~~~~
- Improved the configuration paths handling (#85)
  - Can use all the "." + something (i.e: ., ./dir, ../, ./../, ../../something, etc) as paths
  - Added raises to avoid errors
  - Manifest not mandatory anymore (if no anchors in the config)
  - Config strings resolved as paths only if they contain $ or start by .

Improvements:
~~~~~~~~~~~~~~
- Improved circuit validation for virtual nodes (#86)
  - "components" is mandatory by the validation only if the circuit contains nodes other than virtual nodes

Bug Fixes
~~~~~~~~~
- Fixed error when sampling an empty group in NodePopulation.ids (#83)


Version v0.6.0
---------------

Improvements:
~~~~~~~~~~~~~~
- Propagated changes from the new libsonata.ElementReport API (#62)
- Bumped the libsonata version to 0.1.4 (#62)
- Generalized multiple sonata groups validation of edges and nodes (#79)
- Adapted validation to the sonata original repository examples (#81)
- Improved validation for edge_group_id, edge_group_index and node_population edge's attributes (#82)


Version v0.5.3
--------------

New Features
~~~~~~~~~~~~~

- Added '$node_set' to nodes queries

Improvements:
~~~~~~~~~~~~~~
- Reduced memory usage for fields from @library

Bug Fixes
~~~~~~~~~
- Fixed circuit validation of implicit node ids


Version v0.5.2
--------------

New Features
~~~~~~~~~~~~~
- Added the source/target_in_edges that returns set of edge population names that
  use this node population as source/target

Improvements:
~~~~~~~~~~~~~~
- Checked morphology and model_template fields in both @library or normal group.
- Removed some dependencies to NodePopulation mocks in the different tests

Bug Fixes
~~~~~~~~~
- Removed the mechanisms_dir as a mandatory directory for the circuit validation


Version v0.5.1
--------------

New Features
~~~~~~~~~~~~~
- Added source/target node ids to the available properties for edges

Improvements:
~~~~~~~~~~~~~~
- Checked if a node population contains biophysical nodes before calling .morph
- Improved testing for the morph.py module (removed unneeded mocks)

Bug Fixes
~~~~~~~~~
- Fixed circuit validation when edge_group_id/index are missing (allow missing edge_group_id/index
  for single group population)
- Fixed circuit validation when model_type is part of @library


Version v0.5.0
--------------

New Features
~~~~~~~~~~~~~
- Added the FilteredFrameReport and FilteredSpikeReport classes used as lazy and cached results for
  simulation queries.
- Added plots to the filtered spike/frame reports

Improvements:
~~~~~~~~~~~~~~
- Added the filtered class for the spike and frame reports
- Used categoritical values for attr in @library

Bug Fixes
~~~~~~~~~
- Fixed empty dict / array for reports query
- Fixed edge iter_connection with unique_node_ids


Version v0.4.1
--------------

Bug Fixes
~~~~~~~~~
- Fixed the empty list/array/dict in simulation reports and in node.ids()


Version v0.4.0
--------------

New Features
~~~~~~~~~~~~~
- Added complete support of the node sets
- Added population and node_id keys in node sets and node's queries
- Added the $and and $or operators to the node's queries

Improvements:
~~~~~~~~~~~~~~
- Added node sets class
- Added support for compound node sets in the node sets files
- Added the node_sets_file in the circuit_config and remove it from the node storage


Version v0.3.0
--------------

New Features
~~~~~~~~~~~~~
- Added the Simulation support
  - Simulation config support
  - Spike reports support
  - Frame reports support


Version v0.2.0
--------------

New Features
~~~~~~~~~~~~
- Added the multi-population support for circuits
- Added a sonata circuit validator
- Implement "node_id" in node set files

Improvements:
~~~~~~~~~~~~~~
- Updated the constant containers


Version v0.1.2
--------------

New Features
~~~~~~~~~~~~
- Added "@dynamics:" parameters for edges.

Improvements:
~~~~~~~~~~~~~~
- Always use the node_id naming convention in code docstrings.


Version v0.1.1
--------------

Improvements:
~~~~~~~~~~~~~~
- Run deploy step in Travis only for Python 3.6


Version v0.1.0
--------------

New Features
~~~~~~~~~~~~
- Initial commit
