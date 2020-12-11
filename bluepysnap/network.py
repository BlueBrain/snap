# Copyright (c) 2020-2021, EPFL/Blue Brain Project

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
import abc

import pandas as pd
from cached_property import cached_property

from bluepysnap.exceptions import BluepySnapError


class NetworkObject(abc.ABC):
    """Abstract class for the top level NetworkObjects accessor."""

    def __init__(self, circuit):
        """Initialize the top level NetworkObjects accessor."""
        self._circuit = circuit

    @abc.abstractmethod
    def _get_populations(self):
        """Collects the different NetworkObjectPopulation and returns them as a dict."""

    @cached_property
    def _config(self):
        return self._circuit.config

    @cached_property
    def _populations(self):
        return self._get_populations()

    @cached_property
    def population_names(self):
        """Returns all the NetworkObjects population names from the Circuit."""
        return sorted(self._populations)

    @cached_property
    def property_dtypes(self):
        """Returns all the NetworkObjects property dtypes for the Circuit."""
        def _update(d, index, value):
            if d.setdefault(index, value) != value:
                raise BluepySnapError("Same property with different "
                                      "dtype. {}: {}!= {}".format(index, value, d[index]))

        res = dict()
        for pop in self.values():
            for varname, dtype in pop.property_dtypes.iteritems():
                _update(res, varname, dtype)
        return pd.Series(res)

    def keys(self):
        """Returns iterator on the NetworkObjectPopulation names.

        Made to simulate the behavior of a dict.keys().
        """
        return (name for name in self.population_names)

    def values(self):
        """Returns iterator on the NetworkObjectPopulations.

        Made to simulate the behavior of a dict.values().
        """
        return (self[name] for name in self.population_names)

    def items(self):
        """Returns iterator on the tuples (population name, NetworkObjectPopulations).

        Made to simulate the behavior of a dict.items().
        """
        return ((name, self[name]) for name in self.population_names)

    def __getitem__(self, population_name):
        """Access the NetworkObjectPopulation corresponding to the population 'population_name'."""
        try:
            return self._populations[population_name]
        except KeyError:
            raise BluepySnapError("{} not a {} population.".format(population_name, self.__class__))

    def __iter__(self):
        """Allows iteration over the different NetworkObjectPopulation."""
        return iter(self.keys())

    @cached_property
    def size(self):
        """Total number of NetworkObject inside the circuit."""
        return sum(pop.size for pop in self.values())

    @cached_property
    def property_names(self):
        """Returns all the NetworkObject properties present inside the circuit."""
        return set(prop for pop in self.values() for prop in pop.property_names)

    @abc.abstractmethod
    def ids(self, *args, **kwargs):
        """Resolves the ids of the NetworkObject."""

    @abc.abstractmethod
    def get(self, *args, **kwargs):
        """Returns the properties of a the NetworkObject."""
