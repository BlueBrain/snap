import logging
from collections import defaultdict
from functools import partial
from pathlib import Path

from kgforge.core import Resource
from more_itertools import all_equal, always_iterable, first

from bluepysnap.api.entity import Entity

L = logging.getLogger(__name__)


def _get_path(p):
    return Path(p.replace("file://", ""))


class EntityFactory:
    def __init__(self, connector):
        self._connector = connector
        self._function_registry = defaultdict(dict)
        self.register("DetailedCircuit", "snap", open_circuit_snap)
        self.register("DetailedCircuit", "bluepy", open_circuit_bluepy)
        self.register("Simulation", "snap", open_simulation_snap)
        self.register("Simulation", "bluepy", open_simulation_bluepy)
        self.register("Simulation", "bglibpy", open_simulation_bglibpy)
        self.register("MorphologyRelease", "morph-tool", open_morphology_release)
        self.register(
            [
                "DummyMorphology",
                "NeuronMorphology",
                "ReconstructedPatchedCell",
                "ReconstructedWholeBrainCell",
            ],
            "neurom",
            open_morphology_neurom,
        )
        self.register(
            [
                "BrainAtlasRelease",
                "AtlasRelease",
            ],
            "voxcell",
            open_atlas_voxcell,
        )

    def register(self, resource_types, tool, func):
        """Register a tool to open the given resource type.

        Args:
            resource_type (str or list): type(s) of resources handled by tool.
            tool (str): name of the tool.
            func (callable): any callable accepting a resource as parameter.
        """
        for resource_type in always_iterable(resource_types):
            L.info("Registering tool %s for resource type %s", tool, resource_type)
            # The first registered tool for a type will be used as default.
            self._function_registry[resource_type][tool] = func

    def get_registered_types(self):
        """Return the registered resource types."""
        return set(self._function_registry.keys())

    def get_available_tools(self, resource_type):
        """Return the available tools for a given resource type."""
        return list(self._function_registry.get(resource_type, {}))

    def open(self, resource: Resource, tool=None) -> Entity:
        """Open the resource and return an entity (resource, proxy).

        Args:
            resource: resource to be opened.
            tool (str): tool name to be used to open the resource, or None to use the default.

        Returns:
            Entity: entity binding the resource and the opener.
        """
        return Entity(
            resource,
            retriever=self._connector.get_resource_by_id,
            opener=partial(self._open_resource, tool=tool),
        )

    def _open_resource(self, resource, tool=None):
        """Open the resource and return the associated instance."""
        types = resource.type  # type or list of types
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
            return func(resource)
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


def open_circuit_snap(resource):
    import bluepysnap

    config_path = _get_path(resource.circuitBase.url) / "sonata/circuit_config.json"
    return bluepysnap.Circuit(str(config_path))


def open_circuit_bluepy(resource):
    import bluepy

    config_path = _get_path(resource.circuitBase.url) / "CircuitConfig"
    return bluepy.Circuit(str(config_path))


def open_simulation_snap(resource):
    import bluepysnap

    config_path = _get_path(resource.path) / "sonata/simulation_config.json"
    return bluepysnap.Simulation(str(config_path))


def open_simulation_bluepy(resource):
    import bluepy

    config_path = _get_path(resource.path) / "BlueConfig"
    return bluepy.Simulation(str(config_path))


def open_simulation_bglibpy(resource):
    from bglibpy import SSim

    config_path = _get_path(resource.path) / "BlueConfig"
    return SSim(str(config_path))


def open_morphology_release(resource):
    from morph_tool.morphdb import MorphDB

    config_path = _get_path(resource.morphologyIndex.distribution.url)
    return MorphDB.from_neurondb(config_path)


def open_morphology_neurom(resource):
    import neurom

    supported_formats = {"application/swc", "application/h5"}
    unsupported_formats = set()
    for item in always_iterable(resource.distribution):
        encoding_format = getattr(item, "encodingFormat", "").lower()
        if encoding_format in supported_formats and item.type == "DataDownload":
            path = _get_path(item.contentUrl)
            return neurom.load_neuron(path)
        if encoding_format:
            unsupported_formats.add(encoding_format)
    if unsupported_formats:
        raise RuntimeError(f"Unsupported morphology formats: {unsupported_formats}")
    raise RuntimeError("Missing morphology url")


def open_atlas_voxcell(resource):
    from voxcell.nexus.voxelbrain import Atlas

    path = _get_path(resource.distribution.url)
    return Atlas.open(str(path))
