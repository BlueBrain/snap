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

"""Circuit ids."""

import numpy as np
import pandas as pd

from bluepysnap import utils
from bluepysnap.exceptions import BluepySnapError


class CircuitNodeIds:
    """Global Node ids."""
    def __init__(self, index):
        if not isinstance(index, pd.MultiIndex):
            raise BluepySnapError("index must be a pandas.MultiIndex object.")
        index.names = ["population", "node_ids"]
        self.index = index

    @classmethod
    def create_global_ids(cls, populations, population_ids):
        if isinstance(populations, str):
            populations = np.full(len(population_ids), fill_value=populations)
        index = pd.MultiIndex.from_arrays([populations, population_ids])
        return cls(index)

    def _locate(self, population):
        try:
            return self.index.get_locs(utils.ensure_list(population))
        except KeyError:
            return []

    def filter_population(self, population):
        return CircuitNodeIds(self.index[self._locate(population)])

    def get_populations(self):
        return self.index.get_level_values(0).to_numpy()

    def get_ids(self):
        return self.index.get_level_values(1).to_numpy()

    def __repr__(self):
        res = self.index.__repr__()[len("MultiIndex"):]
        return "CircuitNodeIds" + res

    def __str__(self):
        return self.__repr__()

    def to_csv(self, filepath):
        self.index.to_frame(index=False).to_csv(filepath, index=False)

    @classmethod
    def from_csv(cls, filepath):
        return cls(pd.MultiIndex.from_frame(pd.read_csv(filepath)))

