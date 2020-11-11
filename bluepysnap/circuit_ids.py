# Copyright (c) 2020, EPFL/Blue Brain Project

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

"""Circuit ids."""
from collections import namedtuple

import numpy as np
import pandas as pd

import bluepysnap.utils
from bluepysnap.exceptions import BluepySnapError


CircuitNodeId = namedtuple("CircuitNodeId", ("population", "id"))


class CircuitNodeIds(object):
    """High performances CircuitNodeID container.

    This class aims at defining the global node ids for the circuit. The pandas.MultiIndex class
    is used as backend and can be accessed using CircuitNodeIds.index.
    A global circuit node id is the combination of a population and a node ID inside this
    population.
    """
    def __init__(self, index, sort_index=True):
        """Return an instance of CircuitNodeIds.

        Args:
            index (pandas.MultiIndex): a multi index from pandas with the population names as
                first level and node IDs as the second level.
            sort_index (bool): if true sort the index per population and then node IDs.
        """
        if not isinstance(index, pd.MultiIndex):
            raise BluepySnapError("index must be a pandas.MultiIndex object.")
        index.names = ["population", "node_ids"]
        if sort_index:
            # best perf compared to sort_values. Sorted by population and ids.
            index = index.sortlevel()[0]
        self.index = index

    @classmethod
    def from_arrays(cls, populations, population_ids, sort_index=True):
        """Create a set of ids using two arrays population(s) and id(s).

        Args:
            populations (list/numpy.array): a sequence of populations. If the population is a
                string then all the ids will be connected to this population.
            population_ids (list/numpy.array): a sequence of node IDs or a single node ID.
            sort_index (bool): will sort the index if set to True. Otherwise the ordering from the
                user inputs is kept. Sorting the index can result in better performances.

        Returns:
            CircuitNodeIds: a set of global node IDs created via the populations and the node IDs
                provided.
        """
        populations = np.asarray(populations)
        population_ids = np.asarray(population_ids)

        if len(populations) != len(population_ids):
            raise BluepySnapError("populations and population_ids must have the same size. "
                                  "{} != {}".format(len(populations), len(population_ids)))

        index = pd.MultiIndex.from_arrays([populations, population_ids])
        return cls(index, sort_index=sort_index)

    @classmethod
    def from_dict(cls, ids_dictionary):
        """Create a set of ids using a dictionary as input.

        Args:
            ids_dictionary (dict): a dictionary with the population as keys and node IDs as
                values.

        Notes:
            with the support of python 2 we cannot guaranty the ordering so we force the sorting
            of ids.

        Returns:
            CircuitNodeIds: a set of global node IDs created via the provided dictionary.

        Examples:
            >>> CircuitNodeIds.from_dict({"pop1": [0,2,4], "pop2": [1,2,5]})
        """
        populations = np.empty((0,), dtype=str)
        population_ids = np.empty((0,), dtype=np.int64)
        for population, ids in ids_dictionary.items():
            ids = np.asarray(ids)
            population_ids = np.append(population_ids, ids)
            populations = np.append(populations, np.full(ids.shape, fill_value=population))
        index = pd.MultiIndex.from_arrays([populations, population_ids])
        return cls(index, sort_index=True)

    @classmethod
    def from_tuples(cls, circuit_id_list, sort_index=True):
        """Create a set of ids using a list of CircuitNodeId or tuples as input.

        Args:
            circuit_id_list (list): a list of CircuitNodeId or list of tuples.
            sort_index (bool): will sort the index if set to True. Otherwise the ordering from the
                user inputs is kept. Sorting the index can result in better performances.

        Returns:
            CircuitNodeIds: a set of global node IDs created via the provided dictionary.

        Examples:
            >>> CircuitNodeIds.from_tuples([("pop1", 0), ("pop1", 2), ("pop2", 0)])
            >>> CircuitNodeIds.from_tuples([CircuitNodeId("pop1", 0), CircuitNodeId("pop1", 2),
            >>>                             CircuitNodeId("pop2", 0)])
        """
        return cls(pd.MultiIndex.from_tuples(circuit_id_list), sort_index=sort_index)

    def _locate(self, population):
        """Returns the index indices corresponding to a given population.

        Args:
            population (str): the population name you want to locate inside the MultiIndex.

        Returns:
            numpy.array: indices corresponding to the population.
        """
        try:
            return self.index.get_locs(bluepysnap.utils.ensure_list(population))
        except KeyError:
            return []

    def filter_population(self, population, inplace=False):
        """Filter the IDs corresponding to a population.

        Args:
            population (str): the population you want to extract.
            inplace (bool): if set to True. Do the transformation inplace.

        Returns:
            CircuitNodeIds : a filtered CircuitNodeIds containing only IDs for the given population.
        """
        if not inplace:
            return CircuitNodeIds(self.index[self._locate(population)], sort_index=False)
        self.index = self.index[self._locate(population)]
        return None

    def get_populations(self, unique=False):
        """Returns all population values from the circuit node IDs."""
        if unique:
            return self.index.levels[0].to_numpy()
        return self.index.get_level_values(0).to_numpy()

    def get_ids(self, unique=False):
        """Returns all the ID values from the circuit node IDs."""
        if unique:
            return self.index.levels[1].to_numpy()
        return self.index.get_level_values(1).to_numpy()

    def copy(self):
        """Copy a CircuitNodeIds."""
        return CircuitNodeIds(self.index.copy())

    def sort(self, inplace=False):
        """Sort the CircuitNodeIds by population and then by ids.

        Args:
            inplace (bool): if set to True. Do the transformation inplace.

        Notes:
            sorting a CircuitNodeIds will gives better perf for the population extraction and the
            data extraction from a sorted dataframe.
        """
        if not inplace:
            return CircuitNodeIds(self.index, sort_index=True)
        self.index = self.index.sortlevel()[0]
        return None

    def append(self, other, inplace=False):
        """Append a CircuitNodeIds to the current one.

        Args:
            other (CircuitNodeIds): the other CircuitNodeIds to append to the current one.
            inplace (bool): if set to True. Do the transformation inplace.
        """
        if not inplace:
            return CircuitNodeIds(self.index.append(other.index))
        self.index = self.index.append(other.index)
        return None

    def _slice_index(self, my_slice, inplace=False):
        """Index slicer."""
        res = self if inplace else self.copy()
        res.index = res.index[my_slice]
        if not inplace:
            return res
        return None

    def sample(self, sample_size, inplace=False):
        """Sample a CircuitNodeIds.

        Notes:
            this function takes randomly ``sample_size`` values of a circuit node ids.

        Args:
            sample_size (int): the size of the sample. If the size of the sample is greater than
                the size of the CircuitNodeIds then all ids are taken and shuffled.
            inplace (bool): if set to True. Do the transformation inplace.
        """
        indices = np.random.choice(len(self), size=min(sample_size, len(self)))
        return self._slice_index(indices, inplace=inplace)

    def limit(self, limit_size, inplace=False):
        """Limit the size of a CircuitNodeIds.

        Notes:
            this function takes the first ``limit_size`` values of a circuit node ids.

        Args:
            limit_size (int): the size of the sample. If the size of the sample is greater than
                the size of the CircuitNodeIds then all ids are kept.
            inplace (bool): if set to True. Do the transformation inplace.
        """
        return self._slice_index(slice(0, limit_size), inplace=inplace)

    def to_csv(self, filepath):
        """Save CircuitNodeIds to csv format."""
        self.index.to_frame(index=False).to_csv(filepath, index=False)

    @classmethod
    def from_csv(cls, filepath):
        """Load CircuitNodeIds from csv."""
        return cls(pd.MultiIndex.from_frame(pd.read_csv(filepath)))

    def tolist(self):
        """Convert the CircuitNodeIds to a list of tuples."""
        return self.index.tolist()

    def __repr__(self):
        """Correct repr of the CircuitNodeIds."""
        return repr(self.index).replace("MultiIndex", "CircuitNodeIds", 1)

    def __eq__(self, other):
        """Equality for the CircuitNodeIds."""
        if self is other:
            return True
        if not isinstance(other, CircuitNodeIds):
            return False
        return self.index.equals(other.index)

    def __len__(self):
        """Return the length of the CircuitNodeIds."""
        return len(self.index)

    def __iter__(self):
        """Iterator on the CircuitNodeIds."""
        for index in self.index:
            yield CircuitNodeId(*index)

    def __call__(self, *args, **kwargs):
        """Allows to use the CircuitNodeIds as normal indices in a DataFrame."""
        return self.index
