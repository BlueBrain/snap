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
import numpy as np
import pandas as pd
import six

from bluepysnap import utils
from bluepysnap.exceptions import BluepySnapError


class CircuitNodeIds:
    """Global Node ids.

    This class aims at defining the global node ids for the circuit. The pandas.MultiIndex class
    is used as backend and can be accessed using CircuitNodeIds.index.
    A global circuit node id is the combination of a population and a node ID inside this
    population.
    """
    def __init__(self, index):
        if not isinstance(index, pd.MultiIndex):
            raise BluepySnapError("index must be a pandas.MultiIndex object.")
        index.names = ["population", "node_ids"]
        self.index = index

    @classmethod
    def create_global_ids(cls, populations, population_ids):
        """Create a set of ids using population(s) and ids.

        Args:
            populations (str/list/numpy.array): a sequence of populations. If the population is a
                string then all the ids will be connected to this population.
            population_ids (int/list/numpy.array): a sequence of node IDs or a single node ID.

        Returns:
            CircuitNodeIds: a set of global node IDs created via the populations and the node IDs
                provided.
        """
        def _reformat(to_reformat, other):
            if len(to_reformat) == 1:
                return np.full(len(other), fill_value=to_reformat[0])
            return to_reformat

        if np.issubdtype(type(population_ids), np.integer):
            population_ids = utils.ensure_list(population_ids)

        if isinstance(populations, six.string_types):
            populations = utils.ensure_list(populations)

        populations = _reformat(populations, population_ids)
        population_ids = _reformat(population_ids, populations)

        if len(populations) != len(population_ids):
            raise BluepySnapError("populations and population_ids must have the same size or "
                                  "having a single value. {} != {}".format(len(populations),
                                                                           len(population_ids)))

        index = pd.MultiIndex.from_arrays([populations, population_ids])
        return cls(index)

    def _locate(self, population):
        """Returns the index indices corresponding to a given population.

        Args:
            population (str): the population name you want to locate inside the MultiIndex.

        Returns:
            numpy.array: indices corresponding to the population.
        """
        try:
            return self.index.get_locs(utils.ensure_list(population))
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
            return CircuitNodeIds(self.index[self._locate(population)])
        self.index = self.index[self._locate(population)]

    def get_populations(self, unique=False):
        """Returns all population values from the circuit node IDs."""
        res = self.index.get_level_values(0).to_numpy()
        return np.unique(res) if unique else res

    def get_ids(self, unique=False):
        """Returns all the ID values from the circuit node IDs."""
        res = self.index.get_level_values(1).to_numpy()
        return np.unique(res) if unique else res

    def copy(self):
        """Copy a CircuitNodeIds."""
        return CircuitNodeIds(self.index.copy())

    def append(self, other, inplace=False):
        """Append a NodeCircuitIds to the current one.

        Args:
            other (CircuitNodeIds): the other CircuitNodeIds to append to the current one.
            inplace (bool): if set to True. Do the transformation inplace.
        """
        if not inplace:
            return CircuitNodeIds(self.index.append(other.index))
        self.index = self.index.append(other.index)

    def sample(self, sample_size, inplace=False):
        """Sample a CircuitNodeIds.

        Args:
            sample_size (int): the size of the sample. If the size of the sample is greater than
                the size of the CircuitNodeIds then all ids are taken and shuffled.
            inplace (bool): if set to True. Do the transformation inplace.
        """
        res = self if inplace else self.copy()
        if len(res) != 0:
            sample_size = sample_size if sample_size < len(res) else len(res)
            res.index = res.index[np.random.choice(len(res), size=sample_size, replace=False)]
        if not inplace:
            return res

    def limit(self, limit_size, inplace=False):
        """Sample a CircuitNodeIds.

        Args:
            limit_size (int): the size of the sample. If the size of the sample is greater than
                the size of the CircuitNodeIds then all ids are taken and shuffled.
            inplace (bool): if set to True. Do the transformation inplace.
        """
        res = self if inplace else self.copy()
        res.index = res.index[0: limit_size]
        if not inplace:
            return res

    def to_csv(self, filepath):
        """Save NodeCircuitIds to csv format."""
        self.index.to_frame(index=False).to_csv(filepath, index=False)

    @classmethod
    def from_csv(cls, filepath):
        """Load NodeCircuitIds from csv."""
        return cls(pd.MultiIndex.from_frame(pd.read_csv(filepath)))

    def __repr__(self):
        """Correct repr of the IDs."""
        res = self.index.__repr__()[len("MultiIndex"):]
        return "CircuitNodeIds" + res

    def __str__(self):
        """Correct str of the IDs."""
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, CircuitNodeIds):
            return False
        return self.index.equals(other.index)

    def __len__(self):
        return len(self.index)


