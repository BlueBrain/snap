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
"""Exceptions used throughout the library."""


class BluepySnapError(Exception):
    """Base SNAP exception."""


class BluepySnapDeprecationError(Exception):
    """SNAP deprecation exception."""


class BluepySnapDeprecationWarning(DeprecationWarning):
    """SNAP deprecation warning."""


class BluepySnapValidationError:
    """Error used for reporting of validation errors."""

    FATAL = "FATAL"
    WARNING = "WARNING"
    INFO = "INFO"

    def __init__(self, level, message=None):
        """Error.

        Args:
            level (str): error level
            message (str|None): message
        """
        self.level = level
        self.message = message

    def __str__(self):
        """Returns only message by default."""
        return str(self.message)

    __repr__ = __str__

    def __eq__(self, other):
        """Two errors are equal if inherit from Error and their level, message are equal."""
        if not isinstance(other, BluepySnapValidationError):
            return False
        return self.level == other.level and self.message == other.message

    def __hash__(self):
        """Hash. Errors with the same level and message give the same hash."""
        return hash(self.level) ^ hash(self.message)

    @classmethod
    def warning(cls, message):
        """Shortcut for a warning.

        Args:
            message (str): text message

        Returns:
            Error: Error with level WARNING
        """
        return cls(cls.WARNING, message)

    @classmethod
    def fatal(cls, message):
        """Shortcut for a fatal error.

        Args:
            message (str): text message

        Returns:
            Error: Error with level FATAL
        """
        return cls(cls.FATAL, message)
