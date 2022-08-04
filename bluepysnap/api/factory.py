"""Functions and classes used for instantiating Nexus resources."""
import logging
import os
from collections import defaultdict
from functools import partial
from pathlib import Path

from kgforge.core import Resource
from more_itertools import all_equal, always_iterable, first

from bluepysnap.api.entity import DOWNLOADED_CONTENT_PATH, Entity

L = logging.getLogger(__name__)


def _get_path(p):
    return Path(p.replace("file://", ""))


def _get_path_for_item(item, entity):
    if hasattr(item, "atLocation") and hasattr(item.atLocation, "location"):
        path = _get_path(item.atLocation.location)
        if os.access(path, os.R_OK):
            return path
    if hasattr(item, "contentUrl"):
        entity.download(items=item, path=DOWNLOADED_CONTENT_PATH)
        path = DOWNLOADED_CONTENT_PATH / item.name
        return path

    return None


class EntityFactory:
    """Factory class for instantiating Nexus resources."""

    def __init__(self, api, connector):
        """Initializes EntityFactory.

        Args:
            api (bluepysnap.api.Api): Api instance
            connector (bluepysnap.api.connector.Connector): connector instance
        """
        self._api = api
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
                "NeuronMorphology",
                "ReconstructedCell",
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
        self.register("EModelConfiguration", "custom-wrapper", open_emodelconfiguration)

    def register(self, resource_types, tool, func):
        """Register a tool to open the given resource type.

        Args:
            resource_types (str or list): type(s) of resources handled by tool.
            tool (str): name of the tool.
            func (callable): any callable accepting an entity as parameter.
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

    def open(self, resource: Resource, tool=None):
        """Open the resource and return an entity (resource, proxy).

        Args:
            resource: resource to be opened.
            tool (str): tool name to be used to open the resource, or None to use the default.

        Returns:
            Entity: entity binding the resource and the opener.
        """
        return Entity(
            resource,
            api=self._api,
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
            if func is open_emodelconfiguration:
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


def open_circuit_snap(entity):
    """Open SNAP circuit."""
    import bluepysnap

    if hasattr(entity, "circuitConfigPath"):
        config_path = _get_path(entity.circuitConfigPath.url)
    else:
        # TODO: we should abstain from any hard coded paths (even if partial).
        # This was used for a demo (can also be seen elsewhere). The config path should be a
        # property of the resource.
        config_path = _get_path(entity.circuitBase.url) / "sonata/circuit_config.json"
    return bluepysnap.Circuit(str(config_path))


def open_circuit_bluepy(entity):  # pragma: no cover
    """Open bluepy circuit."""
    import bluepy  # pylint: disable=import-error

    config_path = _get_path(entity.circuitBase.url) / "CircuitConfig"
    return bluepy.Circuit(str(config_path))


def open_simulation_snap(entity):
    """Open SNAP simulation."""
    import bluepysnap

    # TODO: Same as with open_circuit_snap: should abstain from hard coded paths
    config_path = _get_path(entity.path) / "sonata/simulation_config.json"
    return bluepysnap.Simulation(str(config_path))


def open_simulation_bluepy(entity):  # pragma: no cover
    """Open bluepy simulation."""
    import bluepy  # pylint: disable=import-error

    config_path = _get_path(entity.path) / "BlueConfig"
    return bluepy.Simulation(str(config_path))


def open_simulation_bglibpy(entity):  # pragma: no cover
    """Open bluepy simulation with bglibpy."""
    from bglibpy import SSim  # pylint: disable=import-error

    config_path = _get_path(entity.path) / "BlueConfig"
    return SSim(str(config_path))


def open_morphology_release(entity):
    """Open morphology release with morph-tool."""
    from morph_tool.morphdb import MorphDB

    config_path = _get_path(entity.morphologyIndex.distribution.url)
    return MorphDB.from_neurondb(str(config_path))


def open_emodelconfiguration(entity, connector):  # pragma: no cover
    """Open emodel configuration."""
    from bluepysnap.api.wrappers import EModelConfiguration

    # TODO: we need the connector here, since the
    # morphology/SubCellularModelScript (mod file) only exists as text;
    # it's not 'connected'/'linked' to anything in nexus

    def _get_entity_by_filter(type_, filter_):
        resources = connector.get_resources(type_, filter_)
        assert len(resources) == 1, f"Wanted 1 entity, got {len(resources)}"
        ret = resources[0]

        def download(path):
            connector.download_resource(ret.distribution, path)
            return Path(path) / ret.distribution.name

        ret.download = download

        return ret

    def _get_named_entity(type_, name):
        return _get_entity_by_filter(type_, {"name": name})

    def _get_distribution(type_, name):
        return _get_entity_by_filter(type_, {"channelDistribution": name})

    def _get_distribution_for_parameter(param):
        if param.location.startswith("distribution_"):
            return _get_distribution(
                "ElectrophysiologyFeatureOptimisationChannelDistribution",
                param.location.split("distribution_")[1],
            )

        return None

    morphology = _get_named_entity("NeuronMorphology", name=entity.morphology.name)
    mod_file = _get_named_entity("SubCellularModelScript", name=entity.mechanisms.name)
    distributions = {p.name: _get_distribution_for_parameter(p) for p in entity.parameters}

    return EModelConfiguration(
        entity.parameters, entity.mechanisms, distributions, morphology, mod_file
    )


def open_morphology_neurom(entity):
    """Open morphology with NeuroM."""
    import neurom

    supported_formats = {"text/plain", "application/swc", "application/h5"}
    unsupported_formats = set()

    for item in always_iterable(entity.distribution):
        if item.type == "DataDownload":
            encoding_format = getattr(item, "encodingFormat", "").lower()
            if encoding_format in supported_formats:
                path = _get_path_for_item(item, entity)
                if path:
                    return neurom.io.utils.load_morphology(path)
            if encoding_format:
                unsupported_formats.add(encoding_format)

    if unsupported_formats:
        raise RuntimeError(f"Unsupported morphology formats: {unsupported_formats}")

    raise RuntimeError("Missing morphology location")


def open_atlas_voxcell(entity):  # pragma: no cover
    """Open atlas with voxcell."""
    from voxcell.nexus.voxelbrain import Atlas  # pylint: disable=import-error

    path = _get_path(entity.distribution.url)
    return Atlas.open(str(path))
