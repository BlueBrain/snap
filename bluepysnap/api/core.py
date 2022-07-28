import logging
from itertools import chain
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from kgforge.core import KnowledgeGraphForge, Resource
from pandas import DataFrame

from bluepysnap import BluepySnapError
from bluepysnap.api.connector import NexusConnector
from bluepysnap.api.entity import Entity
from bluepysnap.api.factory import EntityFactory, _get_path

L = logging.getLogger(__name__)


class Api:
    def __init__(self, *, bucket, token, nexus_config=None, debug=False, **kwargs):
        nexus_config = (
            nexus_config
            or "https://raw.githubusercontent.com/BlueBrain/nexus-forge/master/examples/notebooks/use-cases/prod-forge-nexus.yml"
        )
        self._forge = KnowledgeGraphForge(
            nexus_config, bucket=bucket, token=token, debug=debug, **kwargs
        )
        self._connector = NexusConnector(forge=self._forge, debug=debug)
        self._factory = EntityFactory(connector=self._connector)
        # children APIs
        self._children = {}
        self.add_child_api(CircuitApi(self))
        self.add_child_api(SimulationApi(self))

    def get_entity_by_id(self, *args, tool=None, **kwargs) -> Entity:
        """Retrieve and return a single entity based on the id."""
        resource = self._connector.get_resource_by_id(*args, **kwargs)
        return self._factory.open(resource, tool=tool)

    def get_entities_by_query(self, *args, tool=None, **kwargs) -> List[Entity]:
        """Retrieve and return a list of entities based on a SPARQL query."""
        resources = self._connector.get_resources_by_query(*args, **kwargs)
        return [self._factory.open(r, tool=tool) for r in resources]

    def get_entities(self, *args, tool=None, **kwargs) -> List[Entity]:
        """Retrieve and return a list of entities based on the resource type and a filter.

        Example:
            api.get_entities(
                "DetailedCircuit",
                {"brainLocation.brainRegion.label": "Thalamus"},
                limit=10,
                tool="snap",
            )
        """
        resources = self._connector.get_resources(*args, **kwargs)
        return [self._factory.open(r, tool=tool) for r in resources]

    def as_dataframe(self, data: List[Entity], store_metadata: bool = True, **kwargs) -> DataFrame:
        """Return a pandas dataframe representing the list of entities."""
        data = [e.resource for e in data]
        return self._forge.as_dataframe(data, store_metadata=store_metadata, **kwargs)

    def as_json(
        self, data: Union[Entity, List[Entity]], store_metadata: bool = True, **kwargs
    ) -> Union[Dict, List[Dict]]:
        """Return a dictionary or a list of dictionaries representing the entities."""
        return self._forge.as_json(data.resource, store_metadata=store_metadata, **kwargs)

    def reopen(self, entity, tool=None):
        """Return a new entity to be opened with a different tool."""
        return self._factory.open(entity.resource, tool=tool)

    def add_child_api(self, child):
        if child.name in self._children:
            raise BluepySnapError(f"api already has child api named '{child}'")

        self._children[child.name] = child

    def get_child_api(self, name):
        return self._children.get(name)

    def get_child_api_by_nexus_type(self, nexus_type):
        """Would return (if found) a child api based on nexus type.

        -> No need for the users to know the "names" of the APIs."""


def _get_path(p):
    return Path(p.replace("file://", ""))


# ChildApi could perhaps be merged with API and the "children" should inherit that
# However, the issue is then how to share the forge instance.
class ChildApi:  # should be abstract
    name = None
    nexus_type = None
    default_tool = None

    def __init__(self, api):
        self.api = api
        api.add_child_api(self)

    def open(self, entity, tool=None):
        """Opening an entity directly w/ a Child Api.

        This would imply that entity.instance would be gone."""
        tool = tool or self.default_tool
        self.tools[tool](entity)

    def get(self, filters=None):
        filters = filters or {}
        return self.api.get_entities(self.nexus_type)

    def download(self, entity):
        """Specific instructions on how to download a certain type of entities, e.g., NeuronMorphology

        Would imply this is passed when the Entity is created:
            Entity(*args, downloader=<child_api>.download, **kwargs)

        And the entity.download would be something like:
            def download(self):
                self._downloader(self)
        """
        pass


"""
EXAMPLE:
    api = API(....)
    circuit_api = self.api.get_child_api('circuit')
    circuit = circuit_api.get()[0]
    simulations = circuit_api.get_simulations_by_circuit(circuit)
"""


class CircuitApi(ChildApi):
    """A mock up of an idea how Child APIs could work."""

    name = "circuit"
    nexus_type = "DetailedCircuit"
    default_tool = "snap"

    def __init__(self, api):
        super().__init__(api)
        self.tools = {"bluepy": self.open_bluepy, "snap": self.open_snap}

    def open_bluepy(self):
        pass

    def open_snap(self, entity):
        import bluepysnap

        if hasattr(entity, "circuitConfigPath"):
            config_path = _get_path(entity.circuitConfigPath.url)
        else:
            # we should abstain from any hard coded paths (even if partial), this was used for a demo
            config_path = _get_path(entity.circuitBase.url) / "sonata/circuit_config.json"
        return bluepysnap.Circuit(str(config_path))

    def get_simulations_by_circuit(self, circuit):
        """Get simulations that used a circuit.

        Args:
            circuit (Entity): circuit

        Returns:
            (list): array of simulations (Entity)

        """
        self.api.get_child_api("simulation").get_simulations_by_circuit(circuit)


class SimulationApi(ChildApi):
    name = "simulation"
    nexus_type = "Simulation"
    default_tool = None

    def __init__(self, api):
        super().__init__(api)
        self.tools = {}

    def get_simulations_by_circuit(self, circuit):
        """Retrieve all simulations that use this circuit either directly or in a campaign."""
        sim1 = self.get({"used": {"id": circuit.id}})

        # get simulation campaigns that used the circuit
        simulation_campaigns = self.api.get_entities(
            "SimulationCampaign", {"used": {"id": circuit.id}}
        )

        # get simulations started by the simulation campaigns
        ids = [r.id for r in simulation_campaigns]
        sim2 = list(
            chain.from_iterable(self.get({"wasStartedBy": {"id": id_}}) for id_ in ids)
        )  # Have to loop as "OR" is not supported by KnowledgeGraphForge.search.

        # merge simulations and remove possible duplicates
        simulations = {s.id: s for s in sim1 + sim2}

        return list(simulations.values())


class MorphologyApi(ChildApi):
    def get_morphologies_from_df(self, df: pd.DataFrame, tool=None) -> List[Entity]:
        """Return morphology entities from a given dataframe of morphologies.

        Args:
            df: dataframe containing at least name and path of the morphology.
            tool: tool to be used to open the morphologies.

        Returns: list of morphology entities.
        """
        df = df[["name", "path"]]
        resources = (
            Resource(
                type="DummyMorphology",
                distribution=[
                    Resource(
                        type="DataDownload",
                        name=name,
                        atLocation=Resource(location=str(path)),
                        encodingFormat=f"application/{path.suffix.lstrip('.')}",
                    )
                ],
            )
            for name, path in df.values
        )
        return [self.api._factory.open(r, tool=tool) for r in resources]

    def get_morphologies_from_morphology_release(
        self, morphology_release: Entity, query, tool=None
    ) -> List[Entity]:
        """Query a morphology release and return morphology entities.

        Args:
            morphology_release: morphology release entity, using MorphDB as instance.
            query: query to filter the morphology release dataframe, using index or the columns:
                name
                mtype
                msubtype
                mtype_no_subtype
                layer
                label
                path
                use_axon
                use_dendrites
                axon_repair
                dendrite_repair
                basal_dendrite_repair
                tuft_dendrite_repair
                oblique_dendrite_repair
                unravel
                use_for_stats
                axon_inputs
            tool: tool to be used to open the morphologies.

        Returns: list of morphology entities.
        """
        # TODO: should we check if it's an instance of MorphDB?
        #       should we use an adapter to support other libraries?
        df = morphology_release.instance.df.query(query)
        return self.get_morphologies_from_df(df, tool=tool)


class ExamplesApi(ChildApi):
    def q1(self):
        # “Give me the last 20 simulations from project ‘SSCxDis’”
        # “Give me the last 20 simulations I executed”
        project = "nse/test"
        user = "ivaska"
        return self.api.get_entities(
            "Simulation",
            {
                "project": project,
                "createdBy": user,
            },
            limit=20,
        )

    def q1b(self):
        # same, but multiple users
        project = "nse/test"
        users = ["ivaska", "other"]
        return self.api.get_entities(
            "Simulation",
            {
                "project": project,
                "createdBy": users,
            },
            limit=20,
        )

    def q2(self):
        # “Give me all the circuits from project==SSCxDis”
        # “Give me the last 10 circuits from the thalamus”
        # “Give me all the circuits from BBP”
        project = "nse/test"
        # label = "primary somatosensory cortex"
        label = "Thalamus"
        return self.api.get_entities(
            "DetailedCircuit",
            {
                "project": project,
                "brainLocation.brainRegion.label": label,
            },
            limit=10,
        )

    def q3(self):
        # retrieve all the morphology releases
        return self.api._connector.get_resources(
            "MorphologyRelease",
            limit=10,
        )

    def q4(self):
        # retrieve all the morphology releases with any of the given names
        names = ["O1-20190624-syn_morph_release", "O1-20190624_morph_release"]
        return self.api._connector.get_resources(
            "MorphologyRelease",
            {"name": names},
            limit=10,
        )

    def q5(self):
        # retrieve the DetailedCircuit(s) using the morphologyRelease with the given name
        name = "O0-20180419_morph_release"
        return self.api._connector.get_resources(
            "DetailedCircuit",
            {"nodeCollection.memodelRelease.morphologyRelease.name": name},
            limit=10,
        )

    def q6(self):
        # retrieve the MorphologyRelease(s) used by the circuit with the given name
        name = "Thalamus microcircuit v1"
        return self.api._connector.get_resources(
            "MorphologyRelease",
            {"^morphologyRelease.^memodelRelease.^nodeCollection.name": name},
            limit=10,
        )

    def q7(self, circuit):
        # retrieve all the simulation campaigns using the given circuit
        return self.api._connector.get_resources(
            "SimulationCampaign",
            {"used": f"<{circuit.id}>"},
            # TODO: alternatively, accept an id without surrounding <> as in
            #  {"used": circuit.id},
            limit=10,
        )
