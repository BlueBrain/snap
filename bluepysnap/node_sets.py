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

import copy
import json

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

    def __init__(self, content, instance):
        """Initializes a node set object from a node sets file.

        Args:
            content (dict): Node sets as a dictionary.
            instance (libsonata.NodeSets): ``libsonata`` node sets instance.

        Returns:
            NodeSets: A NodeSets object.
        """
        self._content = content
        self._instance = instance

    @classmethod
    def from_file(cls, filepath):
        """Create NodeSets instance from a file."""
        content = utils.load_json(filepath)
        instance = libsonata.NodeSets.from_file(filepath)
        return cls(content, instance)

    @classmethod
    def from_string(cls, content):
        """Create NodeSets instance from a JSON string."""
        instance = libsonata.NodeSets(content)
        content = json.loads(content)
        return cls(content, instance)

    @classmethod
    def from_dict(cls, content):
        """Create NodeSets instance from a dict."""
        return cls.from_string(json.dumps(content))

    @property
    def content(self):
        """Access (a copy of) the node sets contents."""
        return copy.deepcopy(self._content)

    @property
    def to_libsonata(self):
        """Libsonata instance of the NodeSets."""
        return self._instance

    def update(self, node_sets):
        """Update the contents of the node set.

        Args:
            node_sets (bluepysnap.NodeSets): The node set to extend this node set with.

        Returns:
            set: Names of any overwritten node sets.
        """
        if isinstance(node_sets, NodeSets):
            overwritten = self._instance.update(node_sets.to_libsonata)
            self._content.update(node_sets.content)
            return overwritten

        raise BluepySnapError(
            f"Unexpected type: '{type(node_sets).__name__}' "
            f"(expected: '{self.__class__.__name__}')"
        )

    def __contains__(self, name):
        """Check if node set exists."""
        if isinstance(name, str):
            return name in self._instance.names

        raise BluepySnapError(f"Unexpected type: '{type(name).__name__}' (expected: 'str')")

    def __getitem__(self, name):
        """Return a node set instance for the given node set name."""
        if name not in self:
            raise BluepySnapError(f"Undefined node set: '{name}'")
        return NodeSet(self._instance, name)

    def __iter__(self):
        """Iter through the different node sets names."""
        return iter(self._instance.names)
