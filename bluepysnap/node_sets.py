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

"""Access to node set data.

For more information see:
https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md#node-sets-file
"""
from collections.abc import Mapping
from copy import deepcopy

import numpy as np
from bluepysnap.exceptions import BluepySnapError
from bluepysnap import utils


def _sanitize(node_set):
    """Sanitize standard node set (not compounds).

    Set a single value instead of a one element list.
    Sorted and unique values for the lists of values.

    Args:
        node_set (Mapping): A standard non compound node set.

    Return:
         map: The sanitized node set.
    """
    for key, values in node_set.items():
        if isinstance(values, list):
            if len(values) == 1:
                node_set[key] = values[0]
            else:
                # sorted unique value list
                node_set[key] = np.unique(np.asarray(values)).tolist()
    return node_set


def _resolve_set(content, resolved, node_set_name):
    """Resolve the node set 'node_set_name' from content.

    The resolved node set is returned and the resolved dict is updated in place with the
    resolved node set.

    Args:
        content (dict): the global dictionary containing all unresolved node sets.
        resolved (dict): the global resolved dictionary containing the already resolved node sets.
        node_set_name (str): the name of the current node set to resolve.

    Returns:
        dict: the resolved node set.

    Notes:
        If the node set is a compound node set then all the sub node sets are also resolved and
        stored inside the resolved dictionary.
    """
    if node_set_name in resolved:
        # return already resolved node_sets
        return resolved[node_set_name]

    # keep the content intact
    set_value = deepcopy(content.get(node_set_name))
    if set_value is None:
        raise BluepySnapError("Missing node_set: '{}'".format(node_set_name))
    if not isinstance(set_value, (Mapping, list)) or not set_value:
        raise BluepySnapError("Ambiguous node_set: '{}'".format({node_set_name: set_value}))
    if isinstance(set_value, Mapping):
        resolved[node_set_name] = _sanitize(set_value)
        return resolved[node_set_name]

    # compounds only
    res = [_resolve_set(content, resolved, sub_set_name) for sub_set_name in set_value]

    resolved[node_set_name] = {"$or": res}
    return resolved[node_set_name]


def _resolve(content):
    """Resolve all node sets in content."""
    resolved = {}
    for set_name in content:
        _resolve_set(content, resolved, set_name)
    return resolved


class NodeSets:
    """Access to node sets data."""

    def __init__(self, filepath):
        """Initializes a node set object from a node sets file.

        Args:
            filepath (str/Path): Path to a SONATA node sets file.

        Returns:
            NodeSets: A NodeSets object.
        """
        self.content = utils.load_json(filepath)
        self.resolved = _resolve(self.content)

    def __getitem__(self, node_set_name):
        """Get the resolved node set using name as key."""
        return self.resolved[node_set_name]

    def __iter__(self):
        """Iter through the different node sets names."""
        return iter(self.resolved)
