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

"""BBP cell / synapse attribute namespace."""

from bluepysnap.sonata_constants import DYNAMICS_PREFIX, ConstContainer, Edge, Node

NODE_TYPES = ['biophysical', 'virtual', 'astrocyte', 'single_compartment', 'point_neuron']
EDGE_TYPES = ['chemical', 'electrical', 'synapse_astrocyte', 'endfoot']


class Astrocyte(ConstContainer):
    """Astrocyte property names."""

    X = "x"  #:
    Y = "y"  #:
    Z = "z"  #:

    RADIUS = "radius"

    MORPHOLOGY = "morphology"  #:
    MORPH_CLASS = "morph_class"
    MODEL_TEMPLATE = "model_template"  #:
    MODEL_TYPE = "model_type"  #:

    MTYPE = "mtype"  #:

    MODEL_TYPE = "model_type"  #:
    MODEL_TEMPLATE = "model_template"  #:
    NODE_TYPE_ID = "node_type_id"


class Biophysical(ConstContainer):
    """Biophysical property names."""

    X = "x"  #:
    Y = "y"  #:
    Z = "z"  #:

    ORIENTATION_W = "orientation_w"  #:
    ORIENTATION_X = "orientation_x"  #:
    ORIENTATION_Y = "orientation_y"  #:
    ORIENTATION_Z = "orientation_z"  #:

    MORPHOLOGY = "morphology"  #:
    MORPH_CLASS = "morph_class"
    MODEL_TEMPLATE = "model_template"  #:
    MODEL_TYPE = "model_type"  #:

    MTYPE = "mtype"  #:
    ETYPE = "etype"  #:

    SYNAPSE_CLASS = "synapse_class"  #:
    HOLDING_CURRENT = DYNAMICS_PREFIX + "holding_current"  #:
    THRESHOLD_CURRENT = DYNAMICS_PREFIX + "threshold_current"  #:
    NODE_TYPE_ID = "node_type_id"


class Vasculature(ConstContainer):
    """Vasculature property names."""

    START_X = "start_x"  #:
    START_Y = "start_y"  #:
    START_Z = "start_z"  #:

    END_X = "end_x"  #:
    END_Y = "end_y"  #:
    END_Z = "end_z"  #:

    START_DIAMETER = "start_diameter"
    END_DIAMETER = "end_diameter"

    START_NODE = "start_node"
    END_NODE = "end_node"

    TYPE = "type"

    SECTION_ID = "section_id"
    SEGMENT_ID = "segment_id"

    MODEL_TYPE = "model_type"  #:
    NODE_TYPE_ID = "node_type_id"


class Chemical(ConstContainer):
    """Chemical connection type property names."""

    # postsynaptic touch position (in the center of the segment)
    POST_X_CENTER = "afferent_center_x"  #:
    POST_Y_CENTER = "afferent_center_y"  #:
    POST_Z_CENTER = "afferent_center_z"  #:

    # postsynaptic touch position (on the segment surface)
    POST_X_SURFACE = "afferent_surface_x"  #:
    POST_Y_SURFACE = "afferent_surface_y"  #:
    POST_Z_SURFACE = "afferent_surface_z"  #:

    POST_SECTION_ID = "afferent_section_id"  #:
    POST_SECTION_POS = "afferent_section_pos"  #:
    POST_SECTION_TYPE = "afferent_section_type"  #:

    POST_SEGMENT_ID = "afferent_segment_id"  #:
    POST_SEGMENT_OFFSET = "afferent_segment_offset"  #:

    # presynaptic touch position (in the center of the segment)
    PRE_X_CENTER = "efferent_center_x"  #:
    PRE_Y_CENTER = "efferent_center_y"  #:
    PRE_Z_CENTER = "efferent_center_z"  #:

    # presynaptic touch position (on the segment surface)
    PRE_X_SURFACE = "efferent_surface_x"  #:
    PRE_Y_SURFACE = "efferent_surface_y"  #:
    PRE_Z_SURFACE = "efferent_surface_z"  #:

    PRE_SECTION_ID = "efferent_section_id"  #:
    PRE_SECTION_POS = "efferent_section_pos"  #:
    PRE_SECTION_TYPE = "efferent_section_type"  #:

    PRE_SEGMENT_ID = "efferent_segment_id"  #:
    PRE_SEGMENT_OFFSET = "efferent_segment_offset"  #:

    G_SYNX = "conductance"  #:
    DTC = "decay_time"  #:
    D_SYN = "depression_time"  #:
    F_SYN = "facilitation_time"  #:
    U_SYN = "u_syn"  #:
    NRRP = "n_rrp_vesicles"  #:
    SPINE_LENGTH = "spine_length"  #:
    TYPE = "syn_type_id"  #:
    AXONAL_DELAY = "delay"  #:

    EDGE_TYPE_ID = "edge_type_id"
    SOURCE_NODE_ID = "@source_node"  #:
    TARGET_NODE_ID = "@target_node"  #:


class GliaGlial(ConstContainer):
    """Chemical connection type property names."""

    # postsynaptic touch position (in the center of the segment)
    POST_X_CENTER = "afferent_center_x"  #:
    POST_Y_CENTER = "afferent_center_y"  #:
    POST_Z_CENTER = "afferent_center_z"  #:

    # postsynaptic touch position (on the segment surface)
    POST_X_SURFACE = "afferent_surface_x"  #:
    POST_Y_SURFACE = "afferent_surface_y"  #:
    POST_Z_SURFACE = "afferent_surface_z"  #:

    POST_SECTION_ID = "afferent_section_id"  #:
    POST_SECTION_POS = "afferent_section_pos"  #:
    POST_SECTION_TYPE = "afferent_section_type"  #:

    POST_SEGMENT_ID = "afferent_segment_id"  #:
    POST_SEGMENT_OFFSET = "afferent_segment_offset"  #:

    # presynaptic touch position (in the center of the segment)
    PRE_X_CENTER = "efferent_center_x"  #:
    PRE_Y_CENTER = "efferent_center_y"  #:
    PRE_Z_CENTER = "efferent_center_z"  #:

    # presynaptic touch position (on the segment surface)
    PRE_X_SURFACE = "efferent_surface_x"  #:
    PRE_Y_SURFACE = "efferent_surface_y"  #:
    PRE_Z_SURFACE = "efferent_surface_z"  #:

    PRE_SECTION_ID = "efferent_section_id"  #:
    PRE_SECTION_POS = "efferent_section_pos"  #:
    PRE_SECTION_TYPE = "efferent_section_type"  #:

    PRE_SEGMENT_ID = "efferent_segment_id"  #:
    PRE_SEGMENT_OFFSET = "efferent_segment_offset"  #:

    SPINE_LENGTH = "spine_length"  #:

    EDGE_TYPE_ID = "edge_type_id"
    SOURCE_NODE_ID = "@source_node"  #:
    TARGET_NODE_ID = "@target_node"  #:


class SynapseAstrocyte(ConstContainer):
    """SynapseAstrocyte connection type property names."""

    ASTRO_SEGMENT_ID = "astrocyte_segment_id"
    ASTRO_SEGMENT_OFFSET = "astrocyte_segment_offset"
    ASTRO_SECTION_ID = "astrocyte_section_id"
    ASTRO_SECTION_POS = "astrocyte_section_pos"

    SYNAPSE_ID = "synapse_id"
    SYNAPSE_POPULATION = "synapse_population"

    EDGE_TYPE_ID = "edge_type_id"
    SOURCE_NODE_ID = "@source_node"  #:
    TARGET_NODE_ID = "@target_node"  #:


class Endfoot(ConstContainer):
    """Endfoot connection type property names."""

    ENDFOOT_ID = "endfoot_id"

    ENDFOOT_X_SURFACE = "endfoot_surface_x"  #:
    ENDFOOT_Y_SURFACE = "endfoot_surface_y"  #:
    ENDFOOT_Z_SURFACE = "endfoot_surface_z"  #:

    VASCULATURE_SECTION_ID = "vasculature_section_id"
    VASCULATURE_SEGMENT_ID = "vasculature_segment_id"
    ASTRO_SECTION_ID = "astrocyte_section_id"

    ENDFOOT_LENGTH = "endfoot_compartment_length"
    ENDFOOT_DIAMETER = "endfoot_compartment_diameter"
    ENDFOOT_PERIMETER = "endfoot_compartment_perimeter"

    EDGE_TYPE_ID = "edge_type_id"
    SOURCE_NODE_ID = "@source_node"  #:
    TARGET_NODE_ID = "@target_node"  #:


class Cell(Node):
    """Cell property names."""

    ME_COMBO = "me_combo"  #:
    MTYPE = "mtype"  #:
    ETYPE = "etype"  #:
    LAYER = "layer"  #:
    REGION = "region"  #:
    SYNAPSE_CLASS = "synapse_class"  #:
    HOLDING_CURRENT = DYNAMICS_PREFIX + 'holding_current'  #:
    THRESHOLD_CURRENT = DYNAMICS_PREFIX + 'threshold_current'  #:


class Synapse(Edge):
    """Synapse property names."""

    PRE_GID = Edge.SOURCE_NODE_ID  #:
    POST_GID = Edge.TARGET_NODE_ID  #:

    D_SYN = "depression_time"  #:
    DTC = "decay_time"  #:
    F_SYN = "facilitation_time"  #:
    G_SYNX = "conductance"  #:
    NRRP = "NRRP"  #:
    TYPE = "syn_type_id"  #:
    U_SYN = "u_syn"  #:
    SPINE_LENGTH = "spine_length"  #:

    PRE_SEGMENT_ID = "efferent_segment_id"  #:
    PRE_SEGMENT_OFFSET = "efferent_segment_offset"  #:
    PRE_MORPH_ID = "efferent_morphology_id"  #:

    POST_SEGMENT_ID = "afferent_segment_id"  #:
    POST_SEGMENT_OFFSET = "afferent_segment_offset"  #:
    POST_BRANCH_TYPE = "afferent_section_type"  #:
