import logging
import os.path
from collections import defaultdict

import bluepy

import bluepysnap
from bluepysnap.api.entity import Entity

L = logging.getLogger(__name__)


class EntityFactory:
    def __init__(self):
        self._function_registry = defaultdict(dict)
        self.register("DetailedCircuit", "snap", self.open_circuit_snap)
        self.register("DetailedCircuit", "bluepy", self.open_circuit_bluepy)
        # self.register("Simulation", "snap", self.open_simulation_snap)
        self.register("Simulation", "bluepy", self.open_simulation_bluepy)

    def register(self, resource_type, tool, func):
        """Register a tool to open the given resource type.

        Args:
            resource_type (str): type of the resource that the tool should be able to handle.
            tool (str): name of the tool.
            func (callable): any callable accepting a resource as parameter.
        """
        self._function_registry[resource_type][tool] = func

    def open(self, resource, tool=None):
        result = None
        tool_functions = self._function_registry[resource.type]
        if not tool_functions:
            raise Exception(f"No available tools to open {resource.type}")
        if tool is None:
            # try all the available tools for the type of resource
            for tool, func in tool_functions.items():
                L.info("Trying to use %s to open %s", tool, resource.type)
                result = func(resource)
                if result is not None:
                    break
        elif tool in tool_functions:
            L.info("Using %s to open %s", tool, resource.type)
            func = tool_functions[tool]
            result = func(resource)
        else:
            raise Exception(f"Tool {tool} not found for {resource.type}")
        if result is None:
            raise Exception(f"Unable to open {resource.type}")
        return Entity(resource, result)

    def open_circuit_snap(self, resource):
        base_path = resource.circuitBase.url.replace("file://", "")
        config_path = os.path.join(base_path, "sonata/circuit_config.json")
        if os.path.exists(config_path):
            return bluepysnap.Circuit(config_path)

    def open_circuit_bluepy(self, resource):
        base_path = resource.circuitBase.url.replace("file://", "")
        config_path = os.path.join(base_path, "CircuitConfig")
        if os.path.exists(config_path):
            return bluepy.Circuit(config_path)

    def open_simulation_snap(self, resource):
        raise NotImplementedError

    def open_simulation_bluepy(self, resource):
        base_path = resource.path.replace("file://", "")
        config_path = os.path.join(base_path, "BlueConfig")
        if os.path.exists(config_path):
            return bluepy.Simulation(config_path)
