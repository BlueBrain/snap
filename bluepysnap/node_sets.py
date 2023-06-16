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

import libsonata

from bluepysnap import utils
from bluepysnap.exceptions import BluepySnapError


class NodeSet:
    """Access to single node set."""

    def __init__(self, node_sets, name):
        """Initializes a single node set object.

        Args:
            node_sets (libsonata.NodeSets): libsonata NodeSets instance.
            name (str): name of the node set.

        Returns:
            NodeSet: A NodeSet object.
        """
        self._node_sets = node_sets
        self._name = name

    def get_ids(self, population, raise_missing_property=True):
        """Get the resolved node set as ids."""
        try:
            return self._node_sets.materialize(self._name, population).flatten()
        except libsonata.SonataError as e:
            if not raise_missing_property and "No such attribute" in e.args[0]:
                return []
            raise BluepySnapError(*e.args) from e


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
        self._instance = libsonata.NodeSets.from_file(filepath)

    def __contains__(self, name):
        """Check if node set exists."""
        if isinstance(name, str):
            return name in self._instance.names

        raise BluepySnapError(f"Unexpected type: '{type(name).__name__}' (expected: 'str')")

    def __getitem__(self, name):
        """Return a node set instance for the given node_set name."""
        if name not in self:
            raise BluepySnapError(f"Undefined node set: '{name}'")
        return NodeSet(self._instance, name)

    def __iter__(self):
        """Iter through the different node sets names."""
        return iter(self._instance.names)
