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


def _collect_populations(partial_config, cls):
    result = {}
    for file_config in partial_config:
        storage = cls(file_config)
        for population in storage.population_names:
            if population in result:
                raise BluepySnapError("Duplicated population: '%s'" % population)
            result[population] = storage.population(population)
    return result


class Circuit(object):
    """Access to circuit data."""

    def __init__(self, config):
        """Initializes a circuit object from a SONATA config file.

        Args:
            config (str): Path to a SONATA config file.

        Returns:
            Circuit: A Circuit object.
        """
        self._config = Config(config).resolve()

    @property
    def config(self):
        """Network config dictionary."""
        return self._config

    @cached_property
    def nodes(self):
        """Access to node population(s). See :py:class:`~bluepysnap.nodes.NodePopulation`."""
        return _collect_populations(
            self._config['networks']['nodes'],
            lambda cfg: NodeStorage(cfg, self)
        )

    @cached_property
    def edges(self):
        """Access to edge population(s). See :py:class:`~bluepysnap.edges.EdgePopulation`."""
        return _collect_populations(
            self._config['networks']['edges'],
            lambda cfg: EdgeStorage(cfg, self)
        )
