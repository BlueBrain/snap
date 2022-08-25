# Copyright (c) 2022, EPFL/Blue Brain Project

# This file is part of BlueBrain SNAP library <https://github.com/BlueBrain/snap>

# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License version 3.0 as published
# by the Free Software Foundation.

# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.

# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""Nexus-forge API integration."""
import logging

from kgforge.core import KnowledgeGraphForge

from bluepysnap.nexus.connector import NexusConnector
from bluepysnap.nexus.factory import EntityFactory

L = logging.getLogger(__name__)
NEXUS_CONFIG_DEFAULT = (
    "https://raw.githubusercontent.com/BlueBrain/nexus-forge/"
    "master/examples/notebooks/use-cases/prod-forge-nexus.yml"
)


class NexusHelper:
    """The "main" class for the nexus-forge integration."""

    def __init__(self, bucket, token, nexus_config=None, debug=False, **kwargs):
        """Instantiate a new NexusHelper class.

        Args:
            bucket (str): Name of the bucket to use (as: ``"ORGANIZATON/PROJECT"``).
            token (str): A base64 encoded Nexus access token.
            nexus_config (str): Path to the nexus config to use.
            debug (bool): A flag that enables more verbose output.
            kwargs (dict): See KnowledgeGraphForge.
        """
        self._forge = KnowledgeGraphForge(
            nexus_config or NEXUS_CONFIG_DEFAULT,
            bucket=bucket,
            token=token,
            debug=debug,
            **kwargs,
        )
        self._connector = NexusConnector(forge=self._forge, debug=debug)
        self._factory = EntityFactory(helper=self, connector=self._connector)

    @property
    def factory(self):
        """:py:class:`~bluepysnap.nexus.factory.EntityFactory` instance creating the entities."""
        return self._factory

    def get_entity_by_id(self, resource_id, tool=None, **kwargs):
        """Retrieve and return a single entity based on the id.

        Args:
            resource_id (str): ID of a Nexus resource.
            tool (str): Name of the tool to open the resource with, or None to use the default tool
                        (see :py:class:`~bluepysnap.nexus.factory.EntityFactory.open`).
            kwargs (dict): See KnowledgeGraphForge.retrieve.

        Returns:
            Entity: Desired resource wrapped as an entity.
        """
        resource = self._connector.get_resource_by_id(resource_id, tool=tool, **kwargs)
        return self._factory.open(resource, tool=tool)

    def get_entities_by_query(self, query, tool=None, **kwargs):
        """Retrieve and return a list of entities based on a SPARQL query.

        Args:
            query (str): Query string to be passed to KnowledgeGraphForge.sparql
            tool (str): Name of the tool to open the resource with, or None to use the default tool
                        (see :py:class:`~bluepysnap.nexus.factory.EntityFactory.open`).
            kwargs (dict): See KnowledgeGraphForge.sparql.

        Returns:
            list: An array of found entities (py:class:~bluepysnap.nexus.entity.Entity`).
        """
        resources = self._connector.get_resources_by_query(query, tool=tool, **kwargs)
        return [self._factory.open(r, tool=tool) for r in resources]

    def get_entities(self, type_, filters=None, tool=None, **kwargs):
        """Retrieve and return a list of entities based on the resource type and a filter.

        Args:
            type_ (str): Resource type (e.g., ``"DetailedCircuit"``).
            filters (dict): Search filters to use.
            tool (str): Name of the tool to open the resource with, or None to use the default tool
                        (see :py:class:`~bluepysnap.nexus.factory.EntityFactory.open`).
            kwargs (dict): See KnowledgeGraphForge.search.

        Returns:
            list: An array of found (kgforge.core.Resource) resources.

        Examples:
            >>> helper.get_entities(
            ...     "DetailedCircuit",
            ...     {"brainLocation": {"brainRegion": {"label": "Thalamus"}}},
            ...     tool="snap",
            ...     limit=10)
        """
        resources = self._connector.get_resources(type_, resource_filter=filters, **kwargs)
        return [self._factory.open(r, tool=tool) for r in resources]

    def as_dataframe(self, data, store_metadata=True, **kwargs):
        """Return a pandas dataframe representing the list of entities.

        Args:
            data (list): List of :py:class:`~bluepysnap.nexus.entity.Entity` objects.
            store_metadata(bool): A flag indicating whether or not to include metadata in the
                                  output.
            kwargs (dict): See KnowledgeGraphForge.as_dataframe.

        Returns:
            pandas.DataFrame: A dataframe containing the data of the entity list.
        """
        data = [e.resource for e in data]
        return self._forge.as_dataframe(data, store_metadata=store_metadata, **kwargs)

    def to_dict(self, entity, store_metadata=True, **kwargs):
        """Return a dictionary or a list of dictionaries representing the entities.

        Args:
            entity (Entity): A single entity.
            store_metadata(bool): A flag indicating whether or not to include metadata in the
                                  output.
            kwargs (dict): See KnowledgeGraphForge.as_json.

        Returns:
            dict: A dictionary containing the data of the entity.
        """
        return self._forge.as_json(entity.resource, store_metadata=store_metadata, **kwargs)

    def reopen(self, entity, tool=None):
        """Return a new entity to be opened with a different tool.

        Args:
            entity (Entity): Entity to be opened.
            tool (str): Name of the tool to open the resource with, or None to use the default tool
                        (see :py:class:`~bluepysnap.nexus.factory.EntityFactory.open`).

        Returns:
            Entity: An entity binding the resource and the opener.
        """
        return self._factory.open(entity.resource, tool=tool)
