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

"""Access to circuit data."""

from cached_property import cached_property

from bluepysnap.config import Config
from bluepysnap.nodes import NodeStorage
from bluepysnap.edges import EdgeStorage
from bluepysnap.exceptions import BluepySnapError

from bluepysnap.utils import ensure_list


def _collect_populations(partial_config, cls, select=None):
    result = {}
    if select is not None:
        select = ensure_list(select)
    for file_config in partial_config:
        storage = cls(file_config)
        for population in storage.population_names:
            if select is None or population in select:
                if population in result:
                    raise BluepySnapError("Duplicated population: '%s'" % population)
                result[population] = storage.population(population)
    if select is not None:
        missing = set(select) - set(result.keys())
        if missing:
            raise BluepySnapError("Missing population(s): '%s'" % missing)
    return result


class Circuit(object):
    """Access to circuit data."""

    def __init__(self, config, node_populations=None, edge_populations=None):
        """Initializes a circuit object from a SONATA config file.

        Args:
            config (str): Path to a SONATA config file.
            node_populations (str/list): Name of the node populations used in the circuit.
            edge_populations (str/list): Name of the edge populations used in the circuit.

        Returns:
            Circuit: A Circuit object.
        """
        self._config = Config(config).resolve()
        self._node_populations = node_populations
        self._edge_populations = edge_populations

    @property
    def config(self):
        """Network config dictionary."""
        return self._config

    @cached_property
    def nodes(self):
        """Access to node population(s). See :py:class:`~bluepysnap.nodes.NodePopulation`."""
        return _collect_populations(
            self._config['networks']['nodes'],
            lambda cfg: NodeStorage(cfg, self),
            select=self._node_populations
        )

    @property
    def node_populations(self):
        """Returns the node population names for the circuit."""
        return list(self.nodes)

    @cached_property
    def edges(self):
        """Access to edge population(s). See :py:class:`~bluepysnap.edges.EdgePopulation`."""
        return _collect_populations(
            self._config['networks']['edges'],
            lambda cfg: EdgeStorage(cfg, self),
            select=self._edge_populations
        )

    @property
    def edge_populations(self):
        """Returns the edge population names for the circuit."""
        return list(self.edges)
