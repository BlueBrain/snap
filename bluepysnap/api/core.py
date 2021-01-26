# WIP, it just shows some possibilities.
import logging

from bluepysnap.api.connector import NexusConnector
from bluepysnap.api.factory import EntityFactory

L = logging.getLogger(__name__)


class Api:
    def __init__(self, nexus_config, *, bucket, token, **kwargs):
        self.connector = NexusConnector(nexus_config, bucket=bucket, token=token, **kwargs)
        self.factory = EntityFactory()
        self.circuit = CircuitApi(self)
        self.simulation = SimulationApi(self)

    def get_rentity_by_id(self, *args, tool=None, **kwargs):
        resource = self.connector.get_resource_by_id(*args, **kwargs)
        entity = self.factory.open(resource, tool=tool)
        return Rentity(resource, entity)

    def get_rentities_by_query(self, *args, tool=None, **kwargs):
        resources = self.connector.get_resources_by_query(*args, **kwargs)
        return [Rentity(r, self.factory.open(r, tool=tool)) for r in resources]

    def get_rentities(self, *args, tool=None, **kwargs):
        resources = self.connector.get_resources(*args, **kwargs)
        return [Rentity(r, self.factory.open(r, tool=tool)) for r in resources]

    def q1(self):
        # “Give me the last 20 simulations from project ‘SSCxDis’”
        # “Give me the last 20 simulations I executed”
        project = "nse/test"
        user = "ivaska"
        return self.connector.get_resources(
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
        return self.connector.get_resources(
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
        return self.connector.get_resources(
            "DetailedCircuit",
            {
                "project": project,
                "brainLocation.brainRegion.label": label,
            },
            limit=10,
        )

    def q3(self):
        # retrieve all the morphology releases
        return self.connector.get_resources(
            "MorphologyRelease",
            limit=10,
        )

    def q4(self):
        # retrieve all the morphology releases with any of the given names
        names = ["O1-20190624-syn_morph_release", "O1-20190624_morph_release"]
        return self.connector.get_resources(
            "MorphologyRelease",
            {"name": names},
            limit=10,
        )

    def q5(self):
        # retrieve the DetailedCircuit(s) using the morphologyRelease with the given name
        name = "O0-20180419_morph_release"
        return self.connector.get_resources(
            "DetailedCircuit",
            {"nodeCollection.memodelRelease.morphologyRelease.name": name},
            limit=10,
        )

    def q6(self):
        # retrieve the MorphologyRelease(s) used by the circuit with the given name
        name = "Thalamus microcircuit v1"
        return self.connector.get_resources(
            "MorphologyRelease",
            {"^morphologyRelease.^memodelRelease.^nodeCollection.name": name},
            limit=10,
        )

    def q7(self, circuit_resource):
        # retrieve all the simulation campaigns using the given circuit
        return self.connector.get_resources(
            "SimulationCampaign",
            {"used": f"<{circuit_resource.id}>"},
            # TODO: alternatively, accept an id without surrounding <>.
            #  {"used": circuit_resource.id},
            limit=10,
        )

    def q8(self, circuit_resource):
        # retrieve all simulations that use this circuit either directly or in a campaign
        query = f"""
            SELECT DISTINCT ?id
            WHERE {{
                {{
                    ?id a Simulation ; wasStartedBy ?scid .
                    ?scid a SimulationCampaign ; used <{circuit_resource.id}> .
                }}
            UNION
                {{
                    ?id a Simulation ; used <{circuit_resource.id}> .
                }}
            }}
            """
        return self.connector.get_resources_by_query(query)

    def open_last_circuit(self):
        # retrieve the last created circuit and open it with snap or bluepy
        resources = self.connector.get_resources("DetailedCircuit", limit=1)
        if not resources:
            L.warning("No circuits found.")
            return None
        return self.factory.open(resources[0])


class ChildApi:
    def __init__(self, api):
        self.api = api


class CircuitApi(ChildApi):
    def get_last_circuit(self, tool=None):
        """Retrieve the last created circuit."""
        resources = self.api.get_rentities("DetailedCircuit", limit=1, tool=tool)
        return resources[0] if resources else None

    def get_simulations(self, circuit, tool=None):
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
        return self.api.get_rentities_by_query(query, tool=tool)


class SimulationApi(ChildApi):
    pass


class Rentity:
    def __init__(self, resource, entity):
        self.resource = resource
        self.entity = entity

    def __getattr__(self, name):
        """Try to get an attribute from resource._store_metadata or from the resource itself."""
        _name = "_" + name
        meta = getattr(self.resource, "_store_metadata", None)
        if meta and _name in meta:
            return meta[_name]
        if hasattr(self.resource, name):
            return getattr(self.resource, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
