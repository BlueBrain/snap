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
import logging
from pathlib import Path

from cached_property import cached_property

from bluepysnap.config import CircuitConfig, CircuitConfigStatus
from bluepysnap.edges import Edges
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.node_sets import NodeSets
from bluepysnap.nodes import Nodes

L = logging.getLogger(__name__)


class Circuit:
    """Access to circuit data."""

    def __init__(self, config):
        """Initializes a circuit object from a SONATA config file.

        Args:
            config (str): Path to a SONATA config file.

        Returns:
            Circuit: A Circuit object.
        """
        self._circuit_config_path = str(Path(config).absolute())
        self._config = CircuitConfig.from_config(config)

        if self.partial_config:
            L.info(
                "Loaded PARTIAL circuit config. Functionality may be limited. "
                "It is up to the user to be diligent when accessing properties."
            )

    @property
    def to_libsonata(self):
        """Libsonata instance of the circuit."""
        return self._config.to_libsonata

    @property
    def config(self):
        """Network config dictionary."""
        return self._config.to_dict()

    def get_node_population_config(self, name):
        """Get node population configuration."""
        try:
            return self._config.node_populations[name]
        except KeyError as e:
            raise BluepySnapError(f"Population config not found for node population: {name}") from e

    def get_edge_population_config(self, name):
        """Get edge population configuration."""
        try:
            return self._config.edge_populations[name]
        except KeyError as e:
            raise BluepySnapError(f"Population config not found for edge population: {name}") from e

    @cached_property
    def node_sets(self):
        """Returns the NodeSets object bound to the circuit."""
        path = self.to_libsonata.node_sets_path
        return NodeSets.from_file(path) if path else NodeSets.from_dict({})

    @cached_property
    def nodes(self):
        """Access to node population(s). See :py:class:`~bluepysnap.nodes.Nodes`."""
        return Nodes(self)

    @cached_property
    def edges(self):
        """Access to edge population(s). See :py:class:`~bluepysnap.edges.Edges`."""
        return Edges(self)

    @cached_property
    def partial_config(self):
        """Check partiality of the config."""
        return self._config.status == CircuitConfigStatus.partial

    def __getstate__(self):
        """Make Circuits pickle-able, without storing state of caches."""
        return self._circuit_config_path

    def __setstate__(self, state):
        """Load from pickle state."""
        self.__init__(state)
