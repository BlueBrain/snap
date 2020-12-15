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
import abc


def _word_swapper(doc, source_word, target_word):
    """Swap a word with another in a docstring."""
    if doc is None:
        return doc
    if source_word is None or target_word is None:
        return doc
    return doc.replace(source_word, target_word)


def _copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)."""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


class DocSubstitutionMeta(type):
    """Tool to update an inherited class documentation.

    Notes:
        Using a metaclass is better than decorator to do that due to the returned type being
        incorrect when using a wrapper. Ex with CircuitNodeIds:
        type(CircuitNodeIds)
        <class 'bluepysnap._doctools.DocSubstitutionDecorator.__call__.<locals>.Wrapped'>
        when with metaclass:
        type(CircuitNodeIds)
        <class 'bluepysnap.CircuitNodeIds'>
        It works well with Sphinx also.
    """
    def __new__(mcs, name, parents, attrs, source_word=None, target_word=None):
        """Define the new class to return."""
        original_attrs = attrs.copy()
        for parent in parents:
            # skip classmethod with isfunction if I use also ismethod as a predicate I can have the
            # classmethod docstring changed but then the cls argument is not automatically skipped.
            for fun_name, fun_value in inspect.getmembers(parent, predicate=inspect.isfunction):
                # skip abstract methods. This is fine because we must override them anyway
                try:
                    if fun_name in parent.__abstractmethods__:
                        continue
                except AttributeError:
                    pass
                # skip overrode functions
                if fun_name in original_attrs:
                    continue
                # skip special methods
                if fun_name.startswith("__"):
                    continue
                changed_fun = _copy_func(fun_value)
                changed_fun.__doc__ = _word_swapper(changed_fun.__doc__, source_word, target_word)
                attrs[fun_name] = changed_fun
        # create the class
        obj = super(DocSubstitutionMeta, mcs).__new__(mcs, name, parents, attrs)
        return obj


class AbstractDocSubstitutionMeta(abc.ABCMeta, DocSubstitutionMeta):
    """Mixin class to use with abstract classes.

    It solves the metaclass conflict.
    """
