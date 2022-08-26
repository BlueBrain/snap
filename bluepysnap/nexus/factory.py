# Copyright (c) 2022, EPFL/Blue Brain Project

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

"""Functions and classes used for instantiating Nexus resources."""
import logging
from collections import defaultdict
from functools import partial

from kgforge.core import Resource
from more_itertools import all_equal, always_iterable, first

from bluepysnap.nexus import tools
from bluepysnap.nexus.entity import Entity

L = logging.getLogger(__name__)


class EntityFactory:
    """Factory class for instantiating Nexus resources."""

    def __init__(self, helper, connector):
        """Instantiate a new EntityFactory.

        Args:
            helper (NexusHelper): NexusHelper instance.
            connector (NexusConnector): NexusConnector instance.
        """
        self._helper = helper
        self._connector = connector
        self._function_registry = defaultdict(dict)
        self.register("DetailedCircuit", "snap", tools.open_circuit_snap)
        self.register("DetailedCircuit", "bluepy", tools.open_circuit_bluepy)
        self.register("Simulation", "snap", tools.open_simulation_snap)
        self.register("Simulation", "bluepy", tools.open_simulation_bluepy)
        self.register("Simulation", "bglibpy", tools.open_simulation_bglibpy)
        self.register("MorphologyRelease", "morph-tool", tools.open_morphology_release)
        self.register(
            [
                "NeuronMorphology",
                "ReconstructedCell",
                "ReconstructedPatchedCell",
                "ReconstructedWholeBrainCell",
            ],
            "neurom",
            tools.open_morphology_neurom,
        )
        self.register(
            [
                "BrainAtlasRelease",
                "AtlasRelease",
            ],
            "voxcell",
            tools.open_atlas_voxcell,
        )
        self.register("EModelConfiguration", "custom-wrapper", tools.open_emodelconfiguration)

    def register(self, resource_types, tool, func):
        """Register a tool to open the given resource type.

        Args:
            resource_types (str, list): Type(s) of resources handled by tool.
            tool (str): Name of the tool to register.
            func (callable): Any callable accepting an entity as parameter.
        """
        for resource_type in always_iterable(resource_types):
            L.info("Registering tool %s for resource type %s", tool, resource_type)
            # The first registered tool for a type will be used as default.
            self._function_registry[resource_type][tool] = func

    def get_registered_types(self):
        """Return the registered resource types.

        Returns:
            set: Registered resource types.
        """
        return set(self._function_registry.keys())

    def get_available_tools(self, resource_type):
        """Return the available tools for a given resource type.

        Args:
            resource_type (str): Type of a nexus resource.

        Returns:
            list: Available tool names.
        """
        return list(self._function_registry.get(resource_type, {}))

    def open(self, resource: Resource, tool=None):
        """Open the resource and return an entity (resource, proxy).

        Args:
            resource (kgforge.core.Resource): The resource to be opened.
            tool (str): Name of the tool to open the resource with, or None to use the default tool.

        Returns:
            Entity: An entity binding the resource and the opener.
        """
        return Entity(
            resource,
            helper=self._helper,
            connector=self._connector,
            opener=partial(self._open_entity, tool=tool),
        )

    def _open_entity(self, entity, tool=None):
        """Open the entity and return the associated instance."""
        types = entity.type  # type or list of types
        tool_functions = self._get_tool_functions(types)
        if tool is None:
            tool, func = first(tool_functions.items())
            L.info("Using the default tool %s to open %s", tool, types)
        elif tool in tool_functions:
            func = tool_functions[tool]
            L.info("Using the specified tool %s to open %s", tool, types)
        else:
            raise RuntimeError(f"Tool {tool} not found for {types}")

        try:
            if func is tools.open_emodelconfiguration:
                # TODO: EModelConfiguration in nexus demo/emodel_pipeline
                # only have the morphology.name, and can't be auto downloaded
                return func(entity, self._connector)
            return func(entity)
        except Exception as ex:
            raise RuntimeError(f"Unable to open {types}") from ex

    def _get_tool_functions(self, types):
        """Iterate over types and return the available functions to open the resource."""
        available_tool_functions = {
            resource_type: self._function_registry[resource_type]
            for resource_type in always_iterable(types)
            if resource_type in self._function_registry
        }
        if not available_tool_functions:
            raise RuntimeError(f"No available tools to open {types}")
        if not all_equal(available_tool_functions.values()):
            raise RuntimeError(f"Multiple tools to open {types}")
        # available_tool_functions contains only one item, or all the values are identical,
        # so the first value can be returned
        return first(available_tool_functions.values())
