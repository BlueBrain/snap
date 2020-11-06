# Copyright (c) 2019, EPFL/Blue Brain Project

# This file is part of BlueBrain SNAP library <https://github.com/BlueBrain/snap>

# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License version 3.0 as published
# by the Free Software Foundation.

# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.

# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""Morphology access."""

import os

import numpy as np
import neurom as nm

from bluepysnap.settings import MORPH_CACHE_SIZE
from bluepysnap.sonata_constants import Node
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.circuit_ids import CircuitNodeId


class MorphHelper(object):
    """Collection of morphology-related methods."""

    def __init__(self, morph_dir, population):
        """Initializes a MorphHelper object from a directory path and a NodePopulation object.

        Args:
            morph_dir (str): Path to the directory containing the node morphologies.
            population (NodePopulation): NodePopulation object used to query the nodes.

        Returns:
            MorphHelper: A MorphHelper object.
        """
        self._morph_dir = morph_dir
        self._population = population

        # all nodes from a population must have the same model type
        if not self._is_biophysical(0):
            raise BluepySnapError("Node population does not contain biophysical nodes.")
        self._load = nm.load_neuron
        if MORPH_CACHE_SIZE is not None:
            try:
                from functools import lru_cache
            except ImportError:  # pragma: nocover
                from functools32 import lru_cache
            self._load = lru_cache(maxsize=MORPH_CACHE_SIZE)(self._load)

    def _is_biophysical(self, node_id):
        return self._population.get(node_id, Node.MODEL_TYPE) == "biophysical"

    def get_filepath(self, node_id):
        """Return path to SWC morphology file corresponding to `node_id`.

        Args:
            node_id (int/CircuitNodeId): could be a int or CircuitNodeId.
        """
        name = self._population.get(node_id, Node.MORPHOLOGY)
        if isinstance(node_id, (int, CircuitNodeId)):
            return os.path.join(self._morph_dir, "%s.swc" % name)
        raise BluepySnapError("node_id must be a int or a CircuitNodeId")

    def get(self, node_id, transform=False):
        """Return NeuroM morphology object corresponding to `node_id`.

        Args:
            node_id (int/CircuitNodeId): could be a single int or a CircuitNodeId.
            transform (bool): If `transform` is True, rotate and translate morphology points
                according to `node_id` position in the circuit.
        """
        filepath = self.get_filepath(node_id)
        result = self._load(filepath)
        if transform:
            A_t = self._population.orientations(node_id).transpose()
            B = self._population.positions(node_id).values
            result = result.transform(lambda p: np.dot(p, A_t) + B)
        return result
