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
DEFAULT_NODE_TYPE = "biophysical"
DEFAULT_EDGE_TYPE = "chemical"

NODE_TYPES = {"biophysical", "virtual", "astrocyte", "single_compartment"}
EDGE_TYPES = {"chemical", "electrical", "synapse_astrocyte", "endfoot"}


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
            except AttributeError as e:
                raise BluepySnapError(
                    "Container classes must derive from classes implementing key_set method"
                ) from e
        all_keys.update(
            name
            for name in vars(cls)
            if not name.startswith("_") and name not in ["key_set", "get"]
        )
        return all_keys

    @classmethod
    def get(cls, const_name):
        """Get a constant from a string name."""
        try:
            res = getattr(cls, const_name)
        except AttributeError as e:
            raise BluepySnapError("{cls} does not have a '{const_name}' member") from e
        return res


class Node(ConstContainer):
    """Node property names ordered in the order of appearance in the circuit spec."""

    # biophysical
    X = "x"  #:
    Y = "y"  #:
    Z = "z"  #:

    ROTATION_ANGLE_X = "rotation_angle_xaxis"  #:
    ROTATION_ANGLE_Y = "rotation_angle_yaxis"  #:
    ROTATION_ANGLE_Z = "rotation_angle_zaxis"  #:

    ORIENTATION_W = "orientation_w"  #:
    ORIENTATION_X = "orientation_x"  #:
    ORIENTATION_Y = "orientation_y"  #:
    ORIENTATION_Z = "orientation_z"  #:

    MORPHOLOGY = "morphology"  #:
    LAYER = "layer"  #:
    MODEL_TEMPLATE = "model_template"  #:
    MODEL_TYPE = "model_type"  #:
    MORPH_CLASS = "morph_class"

    ETYPE = "etype"  #:
    MTYPE = "mtype"  #:
    ME_COMBO = "me_combo"  #:

    SYNAPSE_CLASS = "synapse_class"  #:

    REGION = "region"  #:

    THRESHOLD_CURRENT = DYNAMICS_PREFIX + "threshold_current"  #:
    HOLDING_CURRENT = DYNAMICS_PREFIX + "holding_current"  #:
    AIS_SCALER = DYNAMICS_PREFIX + "AIS_scaler"
    INPUT_RESISTANCE = DYNAMICS_PREFIX + "input_resistance"  #:

    EXC_MINI_FREQUENCY = "exc-mini_frequency"
    INH_MINI_FREQUENCY = "inh-mini_frequency"

    HEMISPHERE = "hemisphere"

    # Not found in spec but kept for compatibility
    RECENTER = "recenter"


class Edge(ConstContainer):
    """Edge property names ordered in the order of appearance in the circuit spec."""

    POST_X_CENTER = "afferent_center_x"  #:
    POST_Y_CENTER = "afferent_center_y"  #:
    POST_Z_CENTER = "afferent_center_z"  #:

    POST_X_SURFACE = "afferent_surface_x"  #:
    POST_Y_SURFACE = "afferent_surface_y"  #:
    POST_Z_SURFACE = "afferent_surface_z"  #:

    POST_SECTION_ID = "afferent_section_id"  #:
    POST_SECTION_POS = "afferent_section_pos"  #:
    POST_SECTION_TYPE = POST_BRANCH_TYPE = "afferent_section_type"  #:

    POST_SEGMENT_ID = "afferent_segment_id"  #:
    POST_SEGMENT_OFFSET = "afferent_segment_offset"  #:

    PRE_X_CENTER = "efferent_center_x"  #:
    PRE_Y_CENTER = "efferent_center_y"  #:
    PRE_Z_CENTER = "efferent_center_z"  #:

    PRE_X_SURFACE = "efferent_surface_x"  #:
    PRE_Y_SURFACE = "efferent_surface_y"  #:
    PRE_Z_SURFACE = "efferent_surface_z"  #:

    PRE_SECTION_ID = "efferent_section_id"  #:
    PRE_SECTION_POS = "efferent_section_pos"  #:
    PRE_SECTION_TYPE = PRE_BRANCH_TYPE = "efferent_section_type"  #:

    PRE_SEGMENT_ID = "efferent_segment_id"  #:
    PRE_SEGMENT_OFFSET = "efferent_segment_offset"  #:

    G_SYNX = "conductance"  #:
    DTC = "decay_time"  #:
    D_SYN = "depression_time"  #:
    F_SYN = "facilitation_time"  #:
    U_SYN = "u_syn"  #:
    NRRP = "NRRP"  #:

    SPINE_LENGTH = "spine_length"  #:
    SPINE_MORPH = "spine_morphology"
    SPINE_PSD_ID = "spine_psd_id"
    SPINE_SHARING_ID = "spine_sharing_id"

    CONDUCTANCE_RATIO = "conductance_scale_factor"
    U_HILL_COEFFICIENT = "u_hill_coefficient"

    TYPE = "syn_type_id"  #:
    PROPERTY_RULE = "syn_property_rule"

    AXONAL_DELAY = DELAY = "delay"  #:

    PRE_GID = SOURCE_NODE_ID = "@source_node"  #:
    POST_GID = TARGET_NODE_ID = "@target_node"  #:

    # Not found in spec but kept for compatibility
    PRE_MORPH_ID = "efferent_morphology_id"
    SYN_WEIGHT = "syn_weight"


# Have aliases for backwards compatibility
Cell = Node
Synapse = Edge
