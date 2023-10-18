# Copyright (c) 2023, EPFL/Blue Brain Project

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

"""CircuitIds types."""

from collections import namedtuple

import numpy as np

# dtypes for the different node and edge ids. We are using np.int64 to avoid the infamous
# https://github.com/numpy/numpy/issues/15084 numpy problem. This type needs to be used for
# all returned node or edge ids.
IDS_DTYPE = np.int64

CircuitNodeId = namedtuple("CircuitNodeId", ("population", "id"))
CircuitEdgeId = namedtuple("CircuitEdgeId", ("population", "id"))
