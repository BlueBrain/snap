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

"""Neuron models access."""

from pathlib import Path

from bluepysnap.exceptions import BluepySnapError
from bluepysnap.sonata_constants import Node
from bluepysnap.utils import is_node_id


class NeuronModelsHelper:
    """Collection of neuron models related methods."""

    def __init__(self, properties, population):
        """Constructor.

        Args:
            properties (libsonata.PopulationProperties): properties of the population
            population (NodePopulation): NodePopulation object used to query the nodes.

        Returns:
            NeuronModelsHelper: A NeuronModelsHelper object.
        """
        # all nodes from a population must have the same model type
        if properties.type != "biophysical":
            raise BluepySnapError("Neuron models can be only in biophysical node population.")

        self._properties = properties
        self._population = population

    def get_filepath(self, node_id):
        """Return path to model file corresponding to `node_id`.

        Args:
            node_id (int|CircuitNodeId): node id

        Returns:
            Path: path to the model file of neuron
        """
        if not is_node_id(node_id):
            raise BluepySnapError("node_id must be a int or a CircuitNodeId")
        node = self._population.get(node_id, [Node.MODEL_TYPE, Node.MODEL_TEMPLATE])
        models_dir = self._properties.biophysical_neuron_models_dir

        template = node[Node.MODEL_TEMPLATE]
        assert ":" in template, "Format of 'model_template' must be <schema>:<resource>."
        schema, resource = template.split(":", 1)
        resource = Path(resource).with_suffix(f".{schema}")
        if resource.is_absolute():
            return resource
        return Path(models_dir, resource)
