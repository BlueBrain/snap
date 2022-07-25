"""Nexus-forge API integration."""
import logging

from kgforge.core import KnowledgeGraphForge

from bluepysnap.api.connector import NexusConnector
from bluepysnap.api.factory import EntityFactory

L = logging.getLogger(__name__)
NEXUS_CONFIG_DEFAULT = (
    "https://raw.githubusercontent.com/BlueBrain/nexus-forge/"
    "master/examples/notebooks/use-cases/prod-forge-nexus.yml"
)


class Api:
    """The "main" class for the nexus-forge integration."""

    def __init__(self, bucket, token, nexus_config=None, debug=False, **kwargs):
        """Initializes the Api class.

        Args:
            bucket (str): name of the bucket to use (as: "ORGANIZATON/PROJECT")
            token (str): base64 encoded Nexus access token
            nexus_config (str): Path to the nexus config to use
            debug (bool): enables more verbose output
            kwargs (dict): see kgforge.core.KnowledgeGraphForge
        """
        self._forge = KnowledgeGraphForge(
            nexus_config or NEXUS_CONFIG_DEFAULT,
            bucket=bucket,
            token=token,
            debug=debug,
            **kwargs,
        )
        self._connector = NexusConnector(forge=self._forge, debug=debug)
        self._factory = EntityFactory(connector=self._connector)

    def get_entity_by_id(self, *args, tool=None, **kwargs):
        """Retrieve and return a single entity based on the id."""
        resource = self._connector.get_resource_by_id(*args, **kwargs)
        return self._factory.open(resource, tool=tool)

    def get_entities_by_query(self, *args, tool=None, **kwargs):
        """Retrieve and return a list of entities based on a SPARQL query."""
        resources = self._connector.get_resources_by_query(*args, **kwargs)
        return [self._factory.open(r, tool=tool) for r in resources]

    def get_entities(self, *args, tool=None, **kwargs):
        """Retrieve and return a list of entities based on the resource type and a filter.

        Example:
            api.get_entities(
                "DetailedCircuit",
                {"brainLocation": {"brainRegion": {"label": "Thalamus"}}},
                limit=10,
                tool="snap",
            )
        """
        resources = self._connector.get_resources(*args, **kwargs)
        return [self._factory.open(r, tool=tool) for r in resources]

    def as_dataframe(self, data, store_metadata=True, **kwargs):
        """Return a pandas dataframe representing the list of entities."""
        data = [e.resource for e in data]
        return self._forge.as_dataframe(data, store_metadata=store_metadata, **kwargs)

    def as_json(self, data, store_metadata=True, **kwargs):
        """Return a dictionary or a list of dictionaries representing the entities."""
        return self._forge.as_json(data.resource, store_metadata=store_metadata, **kwargs)

    def reopen(self, entity, tool=None):
        """Return a new entity to be opened with a different tool."""
        return self._factory.open(entity.resource, tool=tool)
