import logging
import os.path
from collections import defaultdict
from functools import partial

import bluepy
import lazy_object_proxy

import bluepysnap
from bluepysnap.api.entity import Entity

L = logging.getLogger(__name__)


def _try_open(func, *args, **kwargs):
    """Executes a function and return None in case of error."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        L.warning("Open error: %s", e)


class EntityFactory:
    def __init__(self):
        self._function_registry = defaultdict(dict)
        self.register("DetailedCircuit", "snap", self.open_circuit_snap)
        self.register("DetailedCircuit", "bluepy", self.open_circuit_bluepy)
        self.register("Simulation", "snap", self.open_simulation_snap)
        self.register("Simulation", "bluepy", self.open_simulation_bluepy)

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
        proxy = lazy_object_proxy.Proxy(partial(self._open_resource, resource, tool=tool))
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
        base_path = resource.circuitBase.url.replace("file://", "")
        config_path = os.path.join(base_path, "sonata/circuit_config.json")
        return bluepysnap.Circuit(config_path)

    def open_circuit_bluepy(self, resource):
        base_path = resource.circuitBase.url.replace("file://", "")
        config_path = os.path.join(base_path, "CircuitConfig")
        return bluepy.Circuit(config_path)

    def open_simulation_snap(self, resource):
        base_path = resource.path.replace("file://", "")
        config_path = os.path.join(base_path, "sonata/simulation_config.json")
        return bluepysnap.Simulation(config_path)

    def open_simulation_bluepy(self, resource):
        base_path = resource.path.replace("file://", "")
        config_path = os.path.join(base_path, "BlueConfig")
        return bluepy.Simulation(config_path)
