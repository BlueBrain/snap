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

import warnings

from bluepysnap import sonata_constants
from bluepysnap.sonata_constants import (  # pylint: disable=unused-import
    EDGE_TYPES,
    NODE_TYPES,
    Cell,
    Synapse,
)
from bluepysnap.utils import Deprecate

with warnings.catch_warnings():
    # Making sure the warning is shown
    warnings.simplefilter("default", DeprecationWarning)
    Deprecate.warn(
        f"'{__name__}' is deprecated and will be removed in future versions. "
        f"Please use '{sonata_constants.__name__}' instead."
    )
