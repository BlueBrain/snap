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

from bluepysnap import edges
from bluepysnap import nodes


class Cell(object):
    """Cell property names."""

    MORPHOLOGY = "morphology"
    ME_COMBO = "me_combo"
    MTYPE = "mtype"
    ETYPE = "etype"
    LAYER = "layer"
    REGION = "region"
    SYNAPSE_CLASS = "synapse_class"
    X = "x"
    Y = "y"
    Z = "z"
    HOLDING_CURRENT = nodes.DYNAMICS_PREFIX + 'holding_current'
    THRESHOLD_CURRENT = nodes.DYNAMICS_PREFIX + 'threshold_current'


class Synapse(object):
    """Synapse property names."""

    PRE_GID = edges.SOURCE_NODE_ID
    POST_GID = edges.TARGET_NODE_ID

    AXONAL_DELAY = "delay"
    D_SYN = "depression_time"
    DTC = "decay_time"
    F_SYN = "facilitation_time"
    G_SYNX = "conductance"
    NRRP = "NRRP"
    TYPE = "syn_type_id"
    U_SYN = "u_syn"

    PRE_SEGMENT_ID = "efferent_segment_id"
    PRE_SEGMENT_OFFSET = "efferent_segment_offset"
    PRE_SECTION_ID = "efferent_section_id"
    PRE_SECTION_POS = "efferent_section_pos"

    POST_SEGMENT_ID = "afferent_segment_id"
    POST_SEGMENT_OFFSET = "afferent_segment_offset"
    POST_BRANCH_TYPE = "afferent_section_type"
    POST_SECTION_ID = "afferent_section_id"
    POST_SECTION_POS = "efferent_section_pos"

    # presynaptic touch position (in the center of the segment)
    PRE_X_CENTER = "efferent_center_x"
    PRE_Y_CENTER = "efferent_center_y"
    PRE_Z_CENTER = "efferent_center_z"

    # presynaptic touch position (on the segment surface)
    PRE_X_SURFACE = "efferent_surface_x"
    PRE_Y_SURFACE = "efferent_surface_y"
    PRE_Z_SURFACE = "efferent_surface_z"

    # postsynaptic touch position (in the center of the segment)
    POST_X_CENTER = "afferent_center_x"
    POST_Y_CENTER = "afferent_center_y"
    POST_Z_CENTER = "afferent_center_z"

    # postsynaptic touch position (on the segment surface)
    POST_X_SURFACE = "afferent_surface_x"
    POST_Y_SURFACE = "afferent_surface_y"
    POST_Z_SURFACE = "afferent_surface_z"
