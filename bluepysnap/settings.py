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

"""Configuration variables."""

import os

# All possible checks enabled / deprecated methods disallowed
STRICT_MODE = False


def str2bool(value):
    """Convert environment variable value to bool."""
    if value is None:
        return False
    else:
        return value.lower() in ('y', 'yes', 'true', '1')


def load_env():
    """Load settings from environment variables."""
    # pylint: disable=global-statement
    if 'BLUESNAP_STRICT_MODE' in os.environ:
        global STRICT_MODE
        STRICT_MODE = str2bool(os.environ['BLUESNAP_STRICT_MODE'])


load_env()
