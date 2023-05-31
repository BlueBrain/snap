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

    def get_ids(self, node_set_name, population, raise_missing_property=True):
        """Get the resolved node set using name as key."""
        try:
            return self._instance.materialize(node_set_name, population).flatten()
        except libsonata.SonataError as e:
            if "No such attribute" in e.args[0]:
                if not raise_missing_property:
                    return []
            raise BluepySnapError(*e.args) from e

    def __iter__(self):
        """Iter through the different node sets names."""
        return iter(self._instance.names)
