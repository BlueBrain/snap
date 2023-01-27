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

"""Spatial index access."""

from typing import Optional, Union

from spatial_index import open_index


class SpatialIndexHelper:
    """Collection of spatial index methods."""

    def __init__(self, spatial_index_dir, accuracy: Optional[str] = None):
        """Initializes a SpatialIndexHelper object from a directory path.

        Args:
            spatial_index_dir (str): Path to the directory containing the spatial index files.
            accuracy(str): Specifies the accuracy with which indexed elements are treated.
            Allowed are either "bounding_box" or "best_effort". Default: "best_effort"

        Returns:
            SpatialIndexHelper: A SpatialIndexHelper object.
        """
        try:
            self._index = open_index(spatial_index_dir)
        except RuntimeError as e:
            raise IOError(f"Could not load spatial index from f{spatial_index_dir}.") from e

        self.accuracy = accuracy

    def sphere_query(self, center, radius, fields: Optional[Union[str, list[str]]] = None):
        """Returns all elements intersecting with the query sphere.

        Args:
            center = (x, y, z): Coordinates of the center of query sphere
            can also be a numpy array of len==3

            radius (int): Radius of query sphere

            fields (str,list[str]): A string or iterable of strings specifying which
            attributes of the index are to be returned.

        Returns:
        TODO

        """
        return self._index.sphere_query(center, radius, fields=fields, accuracy=self.accuracy)

    def box_query(self, corner, opposing_corner, fields: Optional[Union[str, list[str]]] = None):
        """Returns all elements intersecting with the query box.

        corner and opposing_corner must be two opposing corners of the query box.

        Args:
            corner = (x, y, z): Corner coordinates of an Axis Aligned Box
            can also be a numpy array of len==3

            opposing_corner = (x, y, z): Corner coordinates of an Axis Alligned Box
            can also be a numpy array of len==3

            fields (str,list[str]): A string or iterable of strings specifying which
            attributes of the index are to be returned.

        Returns:
        TODO
        """
        return self._index.box_query(corner, opposing_corner, fields=fields, accuracy=self.accuracy)
