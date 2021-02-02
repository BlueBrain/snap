import logging
from collections import defaultdict
from functools import partial
from pathlib import Path

import bluepy
from lazy_object_proxy import Proxy
from morph_tool.morphdb import MorphDB

import bluepysnap
from bluepysnap.api.entity import Entity, ResolvingResource

L = logging.getLogger(__name__)


def _try_open(func, *args, **kwargs):
    """Executes a function and return None in case of error."""
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
        self.register("DetailedCircuit", "snap", self.open_circuit_snap)
        self.register("DetailedCircuit", "bluepy", self.open_circuit_bluepy)
        self.register("Simulation", "snap", self.open_simulation_snap)
        self.register("Simulation", "bluepy", self.open_simulation_bluepy)
        self.register("MorphologyRelease", "morph-tool", self.open_morphology_release)

    def register(self, resource_type, tool, func):
        """Register a tool to open the given resource type.

        Args:
            resource_type (str): type of the resource that the tool should be able to handle.
            tool (str): name of the tool.
            func (callable): any callable accepting a resource as parameter.
        """
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
        tool_functions = self._function_registry[resource.type]
        if not tool_functions:
            raise RuntimeError(f"No available tools to open {resource.type}")
        if tool is None:
            # try all the available tools for the type of resource
            for tool, func in tool_functions.items():
                L.info("Trying to use %s to open %s", tool, resource.type)
                result = _try_open(func, resource)
                if result is not None:
                    break
        elif tool in tool_functions:
            L.info("Using %s to open %s", tool, resource.type)
            func = tool_functions[tool]
            result = _try_open(func, resource)
        else:
            raise RuntimeError(f"Tool {tool} not found for {resource.type}")
        if result is None:
            raise RuntimeError(f"Unable to open {resource.type}")
        return result

    def open_circuit_snap(self, resource):
        config_path = _get_path(resource.circuitBase.url) / "sonata/circuit_config.json"
        return bluepysnap.Circuit(str(config_path))

    def open_circuit_bluepy(self, resource):
        config_path = _get_path(resource.circuitBase.url) / "CircuitConfig"
        return bluepy.Circuit(str(config_path))

    def open_simulation_snap(self, resource):
        config_path = _get_path(resource.path) / "sonata/simulation_config.json"
        return bluepysnap.Simulation(str(config_path))

    def open_simulation_bluepy(self, resource):
        config_path = _get_path(resource.path) / "BlueConfig"
        return bluepy.Simulation(str(config_path))

    def open_morphology_release(self, resource):
        config_path = _get_path(resource.morphologyIndex.distribution.url)
        return MorphDB.from_neurondb(config_path)
