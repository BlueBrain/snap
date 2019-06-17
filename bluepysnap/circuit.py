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

"""
Access to circuit data.
"""

from cached_property import cached_property

from bluepysnap.config import Config
from bluepysnap.nodes import NodePopulation
from bluepysnap.edges import EdgePopulation
from bluepysnap.exceptions import BlueSnapError


def _collect_populations(configs, cls, select=None):
    result = {}
    for cfg in configs:
        population = cls(cfg)
        if population.name in result:
            raise BlueSnapError("Duplicate population: '%s'" % population.name)
        result[population.name] = population
    if select is None:
        return result
    elif select in result:
        return result[select]
    else:
        raise BlueSnapError("No such population: '%s'" % select)


class Circuit(object):
    """ Access to circuit data. """
    def __init__(self, config, node_population=None, edge_population=None):
        self._config = Config(config).resolve()
        self._node_population = node_population
        self._edge_population = edge_population

    @property
    def config(self):
        """ Network config dictionary. """
        return self._config

    @cached_property
    def nodes(self):
        """ Access to node population(s). """
        return _collect_populations(
            self._config['networks']['nodes'],
            lambda cfg: NodePopulation(cfg, self),
            select=self._node_population
        )

    @cached_property
    def edges(self):
        """ Access to edge population(s). """
        return _collect_populations(
            self._config['networks']['edges'],
            lambda cfg: EdgePopulation(cfg, self),
            select=self._edge_population
        )
