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

"""Implementation of Nexus connector for the kgforge API."""
import logging
from pathlib import Path

L = logging.getLogger(__name__)

NEXUS_NAMESPACE = "https://bluebrain.github.io/nexus/vocabulary/"
PROJECTS_NAMESPACE = "https://bbp.epfl.ch/nexus/v1/projects/"
USERS_NAMESPACE = "https://bbp.epfl.ch/nexus/v1/realms/bbp/users/"

# Keys that need to be prefixed with the nexus namespace.
# FIXME: should it be managed automatically in kgforge.core.archetypes.store.rewrite_sparql?
_NEXUS_KEYS = {
    "constrainedBy",
    "createdAt",
    "createdBy",
    "deprecated",
    "incoming",
    "outgoing",
    "project",
    "schemaProject",
    "rev",
    "self",
    "updatedAt",
    "updatedBy",
}

NAMESPACE_MAPPING = {
    "createdBy": USERS_NAMESPACE,
    "updatedBy": USERS_NAMESPACE,
    "project": PROJECTS_NAMESPACE,
}


def build_search_filters(type_, filters):
    """Build search filters in the format expected by nexusforge.

    Args:
        type_ (str): Resource type (e.g., ``"DetailedCircuit"``).
        filters (dict): Search filters to use.

    Returns:
        dict: Search filters in the format expected by nexusforge.
    """
    search_filters = {}

    def add_namespace(key, value):
        return NAMESPACE_MAPPING.get(key, "") + value

    if type_:
        search_filters["type"] = type_

    for k, v in filters.items():
        if k in _NEXUS_KEYS:
            search_filters[f"_{k}"] = {"id": add_namespace(k, v)}
        else:
            search_filters[k] = v

    return search_filters


class NexusConnector:
    """Handles communication with Nexus."""

    def __init__(self, forge, debug=False):
        """Instantiate a new NexusConnector.

        Args:
            forge (KnowledgeGraphForge): A KnowledgeGraphForge instance.
            debug (bool): A flag that enables more verbose output.
        """
        self._forge = forge
        self._debug = debug

    def search(self, type_, filters, **kwargs):
        """Search for resources in Nexus.

        Args:
            type_ (str): Resource type (e.g., ``"DetailedCircuit"``).
            filters (dict): Search filters to use.
            kwargs (dict): See KnowledgeGraphForge.search.

        Returns:
            list: An array of found (kgforge.core.Resource) resources.
        """
        search_filters = build_search_filters(type_, filters)
        kwargs["debug"] = kwargs.get("debug", self._debug)
        kwargs["search_in_graph"] = kwargs.get("search_in_graph", False)

        return self._forge.search(search_filters, **kwargs)

    def query(self, query, **kwargs):
        """Query resources using SparQL as defined in KnowledgeGraphForge.

        Args:
            query (str): Query string to be passed to KnowledgeGraphForge.sparql.
            kwargs (dict): See KnowledgeGraphForge.sparql.

        Returns:
            list: An array of found (kgforge.core.Resource) resources.
        """
        kwargs["debug"] = kwargs.get("debug", self._debug)
        return self._forge.sparql(query, **kwargs)

    def get_resource_by_id(self, resource_id, **kwargs):
        """Fetch a resource based on its ID.

        Args:
            resource_id (str): ID of a Nexus resource.
            kwargs (dict): See KnowledgeGraphForge.retrieve.

        Returns:
            kgforge.core.Resource: Desired resource.
        """
        kwargs["cross_bucket"] = kwargs.get("cross_bucket", True)
        return self._forge.retrieve(resource_id, **kwargs)

    def get_resources_by_query(self, query, **kwargs):
        """Query for resources and fetch them.

        Args:
            query (str): SparQL query string.
            kwargs (dict): See KnowledgeGraphForge.sparql.

        Returns:
            list: An array of found (kgforge.core.Resource) resources.
        """
        result = self.query(query, **kwargs)
        # TODO: execute requests concurrently, or pass a list of ids if possible,
        #  to avoid calling the nexus endpoint for each resource.
        return [self.get_resource_by_id(r.id) for r in result]

    def get_resources(self, resource_type, resource_filter=None, **kwargs):
        """Search for resources and fetch them.

        Args:
            resource_type (str): Resource type (e.g., ``"DetailedCircuit"``).
            resource_filter (dict): Search filters to use.
            kwargs (dict): See KnowledgeGraphForge.search.

        Returns:
            list: An array of found (kgforge.core.Resource) resources.
        """
        resource_filter = resource_filter or {}
        kwargs["limit"] = kwargs.get("limit", 100)
        resources = self.search(resource_type, resource_filter, **kwargs)

        return [self.get_resource_by_id(r.id) for r in resources]

    def download_resource(self, resource, path):
        """Download a resource.

        Args:
            resource (kgforge.core.Resource): A downloadable resource.
            path (str): The path to the directory into which the data is downloaded.
        """
        file_path = Path(path, resource.name)
        if file_path.is_file():
            L.warning("File %s already exists, not downloading...", file_path)
            return
        if resource.type == "DataDownload" and hasattr(resource, "contentUrl"):
            self._forge.download(resource, "contentUrl", path)
        else:
            raise RuntimeError(f"resource {resource.type} can not be downloaded.")
