`bluesnap` is a Python library for accessing BlueBrain circuit models represented in [SONATA](https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md) format.

[![Build Status](https://travis-ci.com/BlueBrain/snap.svg?branch=master)](https://travis-ci.com/BlueBrain/snap)

The main interface class exposed is `Circuit`, which corresponds to the *static* structure of a neural network, i.e.:
 - node positions / properties
 - edge positions / properties
 - detailed morphologies

Most of `bluesnap` methods return [pandas](https://pandas.pydata.org) Series or DataFrames, indexed in a way to facilitate combining data from different sources (i.e. by node or edge IDs).

Among other dependencies, `bluesnap` relies on BBP-provided libraries:
 - [libsonata](https://github.com/BlueBrain/libsonata), for accessing SONATA files
 - [NeuroM](https://github.com/BlueBrain/NeuroM), for accessing detailed morphologies
