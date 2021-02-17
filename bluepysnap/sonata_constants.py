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
"""Module including the sonata node and edge namespaces."""
from bluepysnap.exceptions import BluepySnapError

DYNAMICS_PREFIX = "@dynamics:"


class ConstContainer:
    """Constant Container for snap.

    Notes:
        Allows the creation of hierarchical subclasses such as:

        .. code-block:: pycon

            >>> class Container(ConstContainer):
            >>>     VAR1 = "var1"
            >>> class SubContainer(Container):
            >>>     VAR2 = "var2"
            >>> class SubSubContainer(SubContainer):
            >>>     VAR3 = "var3"
            >>> print(SubSubContainer.key_set()) # To know the accessible variable names
            {"VAR1", "VAR2", "VAR3"}
            >>> for v in SubSubContainer.key_set(): # To get the variable (names, values)
            >>>     print(v, getattr(SubSubContainer, v))
    """

    @classmethod
    def key_set(cls):
        """List all constant members of the class."""
        all_keys = set()
        for base in cls.__bases__:
            if base is object:
                continue
            try:
                all_keys.update(base.key_set())
            except AttributeError:
                raise BluepySnapError(
                    "Container classes must derive from classes implementing key_set method")
        all_keys.update(
            name for name in vars(cls) if
            not name.startswith('_') and name not in ["key_set", "get"])
        return all_keys

    @classmethod
    def get(cls, const_name):
        """Get a constant from a string name."""
        try:
            res = getattr(cls, const_name)
        except AttributeError:
            raise BluepySnapError(
                "{} does not have a '{}' member".format(cls, const_name))
        return res


class Node(ConstContainer):
    """Node property names."""

    X = "x"  #:
    Y = "y"  #:
    Z = "z"  #:

    ORIENTATION_W = "orientation_w"  #:
    ORIENTATION_X = "orientation_x"  #:
    ORIENTATION_Y = "orientation_y"  #:
    ORIENTATION_Z = "orientation_z"  #:

    ROTATION_ANGLE_X = "rotation_angle_xaxis"  #:
    ROTATION_ANGLE_Y = "rotation_angle_yaxis"  #:
    ROTATION_ANGLE_Z = "rotation_angle_zaxis"  #:

    MORPHOLOGY = "morphology"  #:

    RECENTER = "recenter"  #:

    MODEL_TYPE = "model_type"  #:
    MODEL_TEMPLATE = "model_template"  #:


class Edge(ConstContainer):
    """Edge property names."""

    SOURCE_NODE_ID = "@source_node"  #:
    TARGET_NODE_ID = "@target_node"  #:

    AXONAL_DELAY = "delay"  #:
    SYN_WEIGHT = "syn_weight"  #:

    POST_SECTION_ID = "afferent_section_id"  #:
    POST_SECTION_POS = "afferent_section_pos"  #:
    PRE_SECTION_ID = "efferent_section_id"  #:
    PRE_SECTION_POS = "efferent_section_pos"  #:

    # postsynaptic touch position (in the center of the segment)
    POST_X_CENTER = "afferent_center_x"  #:
    POST_Y_CENTER = "afferent_center_y"  #:
    POST_Z_CENTER = "afferent_center_z"  #:

    # postsynaptic touch position (on the segment surface)
    POST_X_SURFACE = "afferent_surface_x"  #:
    POST_Y_SURFACE = "afferent_surface_y"  #:
    POST_Z_SURFACE = "afferent_surface_z"  #:

    # presynaptic touch position (in the center of the segment)
    PRE_X_CENTER = "efferent_center_x"  #:
    PRE_Y_CENTER = "efferent_center_y"  #:
    PRE_Z_CENTER = "efferent_center_z"  #:

    # presynaptic touch position (on the segment surface)
    PRE_X_SURFACE = "efferent_surface_x"  #:
    PRE_Y_SURFACE = "efferent_surface_y"  #:
    PRE_Z_SURFACE = "efferent_surface_z"  #:
