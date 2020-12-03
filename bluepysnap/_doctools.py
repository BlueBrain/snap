# Copyright (c) 2020, EPFL/Blue Brain Project

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
"""Module containing tools related to the documentation and docstrings."""
import inspect


def _word_swapper(doc, source_word, target_word):
    """Swap a word with another in a docstring."""
    if doc is None:
        return doc
    return doc.replace(source_word, target_word)


class DocUpdater:
    """Tool to update a class documentation."""
    def __init__(self, cls):
        """Return an instance of DocUpdater.

        Args:
            cls (class): the class containing the doc string you want to update.
        """
        self.cls = cls

    def replace_all(self, source_word, target_word):
        """Replace source_word with target_word in all method docstrings."""
        for fun_name, fun_value in inspect.getmembers(self.cls, predicate=inspect.isfunction):
            fun_value.__doc__ = _word_swapper(fun_value.__doc__, source_word, target_word)
