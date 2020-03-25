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

"""Access to node set data."""

import collections

import numpy as np
from bluepysnap.exceptions import BluepySnapError
from bluepysnap import utils


def _sanitize(node_set):
    for key, values in node_set.items():
        if isinstance(values, list):
            if len(values) == 1:
                node_set[key] = values[0]
            else:
                # sorted unique value list
                node_set[key] = np.unique(np.asarray(values)).tolist()
    return node_set


class NodeSets:
    """Access to node sets data."""

    def __init__(self, filepath):
        """Initializes a node set object from a node sets file.

        Args:
            filepath (str): Path to a SONATA node sets file.

        Returns:
            NodeSets: A NodeSets object.
        """
        self.content = utils.load_json(filepath)
        self.resolved = self._resolve()

    def _resolve_set(self, resolved, set_name):
        """Resolve a single node set."""
        if set_name in resolved:
            # return already resolved node_sets
            return resolved[set_name]

        set_value = self.content.get(set_name)
        if set_value is None:
            raise BluepySnapError("Missing node_set: '{}'".format(set_name))
        if not isinstance(set_value, (collections.Mapping, list)) or not set_value:
            raise BluepySnapError("Ambiguous node_set: '{}'".format({set_name: set_value}))

        # keep the self.content intact
        set_value = set_value.copy()

        if isinstance(set_value, collections.Mapping):
            return _sanitize(set_value)

        res = []
        for sub_set_name in set_value:
            sub_res_dict = self._resolve_set(resolved, sub_set_name)
            resolved[sub_set_name] = sub_res_dict
            res.append(sub_res_dict)
        return {"$or": res}

    def _resolve(self):
        """Resolve all node sets."""
        resolved = {}
        for set_name in self.content:
            resolved[set_name] = self._resolve_set(resolved, set_name)
        return resolved

    def __getitem__(self, node_set_name):
        """Get the resolved node set from name."""
        return self.resolved[node_set_name]

    def __iter__(self):
        """Iter through the different node sets."""
        return iter(self.resolved)
