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

from bluepysnap.sonata_constants import DYNAMICS_PREFIX, Edge, Node


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
