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

"""Biophys access."""

import os


class BiophysHelper(object):
    """Collection of morphology-related methods."""

    def __init__(self, biophys_dir, nodes):
        """Initializes a BiophysHelper object from a directory path and a NodePopulation object.

        Args:
            biophys_dir (str): Path to the directory containing the node morphologies.
            nodes (NodePopulation): NodePopulation object used to query the nodes.

        Returns:
            BiophysHelper: A BiophysHelper object.
        """
        self._biophys_dir = biophys_dir
        self._nodes = nodes

    def get_filepath(self, node_id):
        """Return path to biophys file corresponding to `node_id`."""
        template = self._nodes.get(node_id, 'model_template')
        assert ':' in template
        schema, resource = template.split(':', 1)
        if os.path.isabs(resource):
            return resource
        return os.path.join(self._biophys_dir, resource)
