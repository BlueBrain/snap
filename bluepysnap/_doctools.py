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
import types
import functools


def _word_swapper(doc, source_word, target_word):
    """Swap a word with another in a docstring."""
    if doc is None:
        return doc
    return doc.replace(source_word, target_word)


def _copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)."""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


# better than decorator to do that due to the returned type being correct with this
# with wrapper <class 'bluepysnap._doctools.DocSubstitutionDecorator.__call__.<locals>.Wrapped'>
# works well with Sphinx also
class DocSubstitutionMeta(type):
    """Tool to update an inheritate class documentation."""
    def __new__(mcs, name, parents, attrs, source_word=None, target_word=None):
        """Define the new class to return."""
        for parent in parents:
            for fun_name, fun_value in inspect.getmembers(parent, predicate=inspect.isfunction):
                # skip special methods
                if fun_name.startswith("__"):
                    continue

                changed_fun = _copy_func(fun_value)
                changed_fun.__doc__ = _word_swapper(changed_fun.__doc__, source_word, target_word)
                attrs[fun_name] = changed_fun

        # create the class
        obj = super(DocSubstitutionMeta, mcs).__new__(mcs, name, parents, attrs)
        return obj
