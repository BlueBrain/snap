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
"""Module containing the Abstract classes for the Network."""
import abc

import pandas as pd
import numpy as np
from cached_property import cached_property


from bluepysnap import utils
from bluepysnap.exceptions import BluepySnapError


class NetworkObject(abc.ABC):
    """Abstract class for the top level NetworkObjects accessor."""

    def __init__(self, circuit):
        """Initialize the top level NetworkObjects accessor."""
        self._circuit = circuit

    def _get_populations(self, cls, config):
        """Collects the different NetworkObjectPopulation and returns them as a dict."""
        res = {}
        for file_config in config:
            storage = cls(file_config, self._circuit)
            for population in storage.population_names:  # pylint: disable=not-an-iterable
                if population in res:
                    raise BluepySnapError(
                        "Duplicated {} population: '{}'".format(self.__class__.__name__,
                                                                population))
                res[population] = storage.population(population)
        return res

    @abc.abstractmethod
    def _collect_populations(self):
        """Should specify the self._get_populations arguments."""

    @cached_property
    def _config(self):
        return self._circuit.config

    @cached_property
    def _populations(self):
        """Cached population dictionary."""
        return self._collect_populations()

    @cached_property
    def population_names(self):
        """Returns all the sorted NetworkObjects population names from the Circuit."""
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

    def _get_ids_from_pop(self, fun_to_apply, returned_ids_cls, sample=None, limit=None):
        """Get CircuitIds of class 'returned_ids_cls' for all populations using 'fun_to_apply'.

        Args:
            fun_to_apply (function): A function that returns the list of IDs for each population
                and the population containing these IDs.
            returned_ids_cls (CircuitNodeIds/CircuitEdgeIds): the class for the CircuitIds.
            sample (int): If specified, randomly choose ``sample`` number of
                IDs from the match result. If the size of the sample is greater than
                the size of all the NetworkObjectPopulation then all ids are taken and shuffled.
            limit (int): If specified, return the first ``limit`` number of
                IDs from the match result. If limit is greater than the size of all the population
                then all IDs are returned.

        Returns:
            CircuitNodeIds/CircuitEdgeIds: containing the IDs and the populations.
        """
        str_type = "<U{}".format(max(len(pop) for pop in self.population_names))
        ids = []
        populations = []
        for pop in self.values():
            pop_ids, name_ids = fun_to_apply(pop)
            pops = np.full_like(pop_ids, fill_value=name_ids, dtype=str_type)
            ids.append(pop_ids)
            populations.append(pops)
        ids = utils.ensure_ids(np.concatenate(ids))
        populations = np.concatenate(populations).astype(str_type)
        res = returned_ids_cls.from_arrays(populations, ids)
        if sample:
            res.sample(sample, inplace=True)
        if limit:
            res.limit(limit, inplace=True)
        return res

    @abc.abstractmethod
    def ids(self, group=None, sample=None, limit=None):
        """Resolves the ids of the NetworkObject."""

    @abc.abstractmethod
    def get(self, group=None, properties=None):
        """Returns the properties of a the NetworkObject."""
        ids = self.ids(group)
        properties = utils.ensure_list(properties)
        # We don t convert to set properties itself to keep the column order.
        properties_set = set(properties)

        unknown_props = properties_set - self.property_names
        if unknown_props:
            raise BluepySnapError("Unknown properties required: {}".format(unknown_props))

        res = pd.DataFrame(index=ids.index, columns=properties)
        for name, pop in self.items():
            global_pop_ids = ids.filter_population(name)
            pop_ids = global_pop_ids.get_ids()
            pop_properties = properties_set & pop.property_names
            # indices from Population and get functions are different so I cannot
            # use a dataframe equal directly and properties have different types so cannot use a
            # multi dim numpy array
            for prop in pop_properties:
                res.loc[global_pop_ids.index, prop] = pop.get(pop_ids, prop).to_numpy()
        return res.sort_index()
