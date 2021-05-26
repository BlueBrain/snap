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

    def __init__(self, config_components, population):
        """Constructor.

        Args:
            config_components (dict): 'components' part of circuit's config
            population (NodePopulation): NodePopulation object used to query the nodes.

        Returns:
            NeuronModelsHelper: A NeuronModelsHelper object.
        """
        # all nodes from a population must have the same model type
        if population.get(0, Node.MODEL_TYPE) not in {"biophysical", "point_neuron"}:
            raise BluepySnapError(
                "Neuron models can be only in biophysical or point node population.")

        self._config_components = config_components
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
        if node[Node.MODEL_TYPE] == "biophysical":
            models_dir = self._config_components.get("biophysical_neuron_models_dir")
            if models_dir is None:
                raise BluepySnapError(
                    "Missing 'biophysical_neuron_models_dir' in Sonata config")
        else:
            models_dir = self._config_components.get("point_neuron_models_dir")
            if models_dir is None:
                raise BluepySnapError("Missing 'point_neuron_models_dir' in Sonata config")

        template = node[Node.MODEL_TEMPLATE]
        assert ':' in template, "Format of 'model_template' must be <schema>:<resource>."
        schema, resource = template.split(':', 1)
        resource = Path(resource).with_suffix(f'.{schema}')
        if resource.is_absolute():
            return resource
        return Path(models_dir, resource)
