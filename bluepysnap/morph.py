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

from pathlib import Path

import numpy as np
from morphio.mut import Morphology
import morph_tool.transform as transformations


from bluepysnap.sonata_constants import Node
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.utils import is_node_id

EXTENSIONS_MAPPING = {
    'asc': 'neurolucida-asc',
    'h5': 'h5v1',
}


class MorphHelper:
    """Collection of morphology-related methods."""

    def __init__(self, morph_dir, population, alternate_morphologies=None):
        """Initializes a MorphHelper object from a directory path and a NodePopulation object.

        Args:
            morph_dir (str): Path to the directory containing the node morphologies.
            population (NodePopulation): NodePopulation object used to query the nodes.
            alternate_morphologies (dict): Dictionary containing paths to alternate morphologies.

        Returns:
            MorphHelper: A MorphHelper object.
        """
        self._morph_dir = morph_dir or ''
        self._alternate_morphologies = alternate_morphologies or {}
        self._population = population

    def _get_morph_dir(self, extension):
        """Return morphology directory based on a given extension."""
        if extension == 'swc':
            if not self._morph_dir:
                raise BluepySnapError("'morphologies_dir' is not defined in config")
            return self._morph_dir

        alternate_key = EXTENSIONS_MAPPING.get(extension)
        if not alternate_key:
            raise BluepySnapError(f"Unsupported extension: {extension}")

        morph_dir = self._alternate_morphologies.get(alternate_key)
        if not morph_dir:
            raise BluepySnapError(f"'{alternate_key}' is not defined in 'alternate_morphologies'")

        return morph_dir

    def get_filepath(self, node_id, extension='swc'):
        """Return path to SWC morphology file corresponding to `node_id`.

        Args:
            node_id (int/CircuitNodeId): could be a int or CircuitNodeId.
            extension (str): expected filetype extension of the morph file.
        """
        if not is_node_id(node_id):
            raise BluepySnapError("node_id must be a int or a CircuitNodeId")
        name = self._population.get(node_id, Node.MORPHOLOGY)

        return Path(self._get_morph_dir(extension), f"{name}.{extension}")

    def get(self, node_id, transform=False, extension='swc'):
        """Return NeuroM morphology object corresponding to `node_id`.

        Args:
            node_id (int/CircuitNodeId): could be a single int or a CircuitNodeId.
            transform (bool): If `transform` is True, rotate and translate morphology points
                according to `node_id` position in the circuit.
            extension (str): expected filetype extension of the morph file.
        """
        filepath = self.get_filepath(node_id, extension=extension)
        result = Morphology(filepath)
        if transform:
            T = np.eye(4)
            T[:3, :3] = self._population.orientations(node_id)  # rotations
            T[:3, 3] = self._population.positions(node_id).values  # translations
            transformations.transform(result, T)
        return result.as_immutable()
