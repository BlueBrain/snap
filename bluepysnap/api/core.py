import logging

from bluepysnap.api.connector import NexusConnector
from bluepysnap.api.factory import EntityFactory

L = logging.getLogger(__name__)


class Api:
    def __init__(self, nexus_config, *, bucket, token, **kwargs):
        self.connector = NexusConnector(nexus_config, bucket=bucket, token=token, **kwargs)
        self.factory = EntityFactory()
        # children APIs
        self.circuit = CircuitApi(self)
        self.simulation = SimulationApi(self)
        self.examples = ExamplesApi(self)

    def get_entity_by_id(self, *args, tool=None, **kwargs):
        resource = self.connector.get_resource_by_id(*args, **kwargs)
        return self.factory.open(resource, tool=tool)

    def get_entities_by_query(self, *args, tool=None, **kwargs):
        resources = self.connector.get_resources_by_query(*args, **kwargs)
        return [self.factory.open(r, tool=tool) for r in resources]

    def get_entities(self, *args, tool=None, **kwargs):
        resources = self.connector.get_resources(*args, **kwargs)
        return [self.factory.open(r, tool=tool) for r in resources]


class ChildApi:
    def __init__(self, api):
        self.api = api


class CircuitApi(ChildApi):
    def get_last_circuit(self, tool=None):
        """Retrieve the last created circuit."""
        entities = self.api.get_entities("DetailedCircuit", limit=1, tool=tool)
        return entities[0] if entities else None


class SimulationApi(ChildApi):
    def get_simulations_by_circuit(self, circuit, tool=None):
        """Retrieve all simulations that use this circuit either directly or in a campaign."""
        query = f"""
            SELECT DISTINCT ?id
            WHERE {{
                {{
                    ?id a Simulation ; wasStartedBy ?scid .
                    ?scid a SimulationCampaign ; used <{circuit.id}> .
                }}
            UNION
                {{
                    ?id a Simulation ; used <{circuit.id}> .
                }}
            }}
            """
        return self.api.get_entities_by_query(query, tool=tool)


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
        return self.api.connector.get_resources(
            "MorphologyRelease",
            limit=10,
        )

    def q4(self):
        # retrieve all the morphology releases with any of the given names
        names = ["O1-20190624-syn_morph_release", "O1-20190624_morph_release"]
        return self.api.connector.get_resources(
            "MorphologyRelease",
            {"name": names},
            limit=10,
        )

    def q5(self):
        # retrieve the DetailedCircuit(s) using the morphologyRelease with the given name
        name = "O0-20180419_morph_release"
        return self.api.connector.get_resources(
            "DetailedCircuit",
            {"nodeCollection.memodelRelease.morphologyRelease.name": name},
            limit=10,
        )

    def q6(self):
        # retrieve the MorphologyRelease(s) used by the circuit with the given name
        name = "Thalamus microcircuit v1"
        return self.api.connector.get_resources(
            "MorphologyRelease",
            {"^morphologyRelease.^memodelRelease.^nodeCollection.name": name},
            limit=10,
        )

    def q7(self, circuit):
        # retrieve all the simulation campaigns using the given circuit
        return self.api.connector.get_resources(
            "SimulationCampaign",
            {"used": f"<{circuit.id}>"},
            # TODO: alternatively, accept an id without surrounding <> as in
            #  {"used": circuit.id},
            limit=10,
        )
