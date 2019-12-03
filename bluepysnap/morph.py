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


class MorphHelper(object):
    """Collection of morphology-related methods."""

    def __init__(self, morph_dir, nodes):
        """Initializes a MorphHelper object from a directory path and a NodePopulation object.

        Args:
            morph_dir (str): Path to the directory containing the node morphologies.
            nodes (NodePopulation): NodePopulation object used to query the nodes.

        Returns:
            MorphHelper: A MorphHelper object.
        """
        self._morph_dir = morph_dir
        self._nodes = nodes
        self._load = nm.load_neuron
        if MORPH_CACHE_SIZE is not None:
            try:
                from functools import lru_cache
            except ImportError:  # pragma: nocover
                from functools32 import lru_cache
            self._load = lru_cache(maxsize=MORPH_CACHE_SIZE)(self._load)

    def get_filepath(self, node_id):
        """Return path to SWC morphology file corresponding to `node_id`."""
        name = self._nodes.get(node_id, Node.MORPHOLOGY)
        return os.path.join(self._morph_dir, "%s.swc" % name)

    def get(self, node_id, transform=False):
        """Return NeuroM morphology object corresponding to `node_id`.

        If `transform` is True, rotate and translate morphology points
        according to `node_id` position in the circuit.
        """
        filepath = self.get_filepath(node_id)
        result = self._load(filepath)
        if transform:
            A_t = self._nodes.orientations(node_id).transpose()
            B = self._nodes.positions(node_id).values
            result = result.transform(lambda p: np.dot(p, A_t) + B)
        return result
