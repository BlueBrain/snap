import logging
from typing import Dict, List, Union

import pandas as pd
from kgforge.core import KnowledgeGraphForge, Resource
from pandas import DataFrame

from bluepysnap.api.connector import NexusConnector
from bluepysnap.api.entity import Entity
from bluepysnap.api.factory import EntityFactory

L = logging.getLogger(__name__)
NEXUS_CONFIG_DEFAULT = (
    "https://raw.githubusercontent.com/BlueBrain/nexus-forge/"
    "master/examples/notebooks/use-cases/prod-forge-nexus.yml"
)


class Api:
    def __init__(self, *, bucket, token, nexus_config=None, debug=False, **kwargs):
        nexus_config = nexus_config or NEXUS_CONFIG_DEFAULT
        self._forge = KnowledgeGraphForge(
            nexus_config, bucket=bucket, token=token, debug=debug, **kwargs
        )
        self._connector = NexusConnector(forge=self._forge, debug=debug)
        self._factory = EntityFactory(connector=self._connector)

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
