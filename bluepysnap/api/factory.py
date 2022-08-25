import logging
from collections import defaultdict
from functools import partial
from pathlib import Path

from lazy_object_proxy import Proxy
from more_itertools import all_equal, always_iterable, first

from bluepysnap.api.entity import Entity, ResolvingResource

L = logging.getLogger(__name__)


def _try_open(func, *args, **kwargs):
    """Execute a function and return None in case of error."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        L.warning("Open error: %s", e)


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

    def register(self, resource_types, tool, func):
        """Register a tool to open the given resource type.

        Args:
            resource_type (str or list): type(s) of resources handled by tool.
            tool (str): name of the tool.
            func (callable): any callable accepting a resource as parameter.
        """
        for resource_type in always_iterable(resource_types):
            L.info("Registering tool %s for resource type %s", tool, resource_type)
            self._function_registry[resource_type][tool] = func

    def open(self, resource, tool=None):
        """Open the resource and return an entity (resource, proxy)."""
        resource = ResolvingResource(resource, retriever=self._connector.get_resource_by_id)
        proxy = Proxy(partial(self._open_resource, resource, tool=tool))
        return Entity(resource, proxy)

    def _open_resource(self, resource, tool=None):
        """Open the resource and return the associated instance."""
        result = None
        types = resource.type
        tool_functions = self._get_tool_functions(types)
        if tool is None:
            # try all the available tools for the type of resource
            for tool, func in tool_functions.items():
                L.info("Trying to use %s to open %s", tool, types)
                result = _try_open(func, resource)
                if result is not None:
                    break
        elif tool in tool_functions:
            L.info("Using %s to open %s", tool, types)
            func = tool_functions[tool]
            result = _try_open(func, resource)
        else:
            raise RuntimeError(f"Tool {tool} not found for {types}")
        if result is None:
            raise RuntimeError(f"Unable to open {types}")
        return result

    def _get_tool_functions(self, types):
        available_tool_functions = {
            t: self._function_registry[t]
            for t in always_iterable(types)
            if t in self._function_registry
        }
        if not available_tool_functions:
            raise RuntimeError(f"No available tools to open {types}")
        if not all_equal(available_tool_functions.values()):
            raise RuntimeError(f"Different tools to open {types}")
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
