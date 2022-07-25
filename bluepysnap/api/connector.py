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
        type_ (str): resource type (e.g., 'DetailedCircuit')
        filters (dict): search filters

    Returns:
        (dict): search filters in the format expected by nexusforge
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
        """Initializes NexusConnector.

        Args:
            forge (kgforge.core.KnowledgeGraphForge): KG Forge instance
            debug (bool): enable more verbose output
        """
        self._forge = forge
        self._debug = debug

    def search(self, type_, filters, **kwargs):
        """Search for resources in Nexus.

        Args:
            type_ (str): resource type (e.g., 'DetailedCircuit')
            filters (dict): search filters
            **kwargs: See KnowledgeGraphForge.search

        Returns:
            (list): an array of found (kgforge.core.Resource) resources
        """
        search_filters = build_search_filters(type_, filters)
        kwargs["debug"] = kwargs.get("debug", self._debug)
        kwargs["search_in_graph"] = kwargs.get("search_in_graph", False)

        return self._forge.search(search_filters, **kwargs)

    def query(self, query, **kwargs):
        """Query resources using SparQL as defined in kgforge.core.KnowledgeGraphForge.

        Args:
            query (str): Query string to be passed to KnowledgeGraphForge.sparql
            **kwargs: See KnowledgeGraphForge.sparql

        Returns:
            (list): an array of found (kgforge.core.Resource) resources
        """
        kwargs["debug"] = kwargs.get("debug", self._debug)
        return self._forge.sparql(query, **kwargs)

    def get_resource_by_id(self, resource_id, **kwargs):
        """Fetch a resource based on its ID.

        Args:
            resource_id (str): ID of a resource
            **kwargs: See KnowledgeGraphForge.retrieve

        Returns:
            (kgforge.core.Resource): desired resource (synchronized)
        """
        kwargs["cross_bucket"] = kwargs.get("cross_bucket", True)
        return self._forge.retrieve(resource_id, **kwargs)

    def get_resources_by_query(self, query, **kwargs):
        """Query for resources and fetch them.

        Args:
            query (str): SparQL query string
            **kwargs: See KnowledgeGraphForge.sparql

        Returns:
            (list): an array of found (kgforge.core.Resource) resources (synchronized)
        """
        result = self.query(query, **kwargs)
        # TODO: execute requests concurrently, or pass a list of ids if possible,
        #  to avoid calling the nexus endpoint for each resource.
        return [self.get_resource_by_id(r.id) for r in result]

    def get_resources(self, resource_type, resource_filter=None, **kwargs):
        """Search for resources and fetch them.

        Args:
            resource_type (str): resource type (e.g., 'DetailedCircuit')
            resource_filter (dict): search filters
            **kwargs: See KnowledgeGraphForge.search

        Returns:
            (list): an array of found (kgforge.core.Resource) resources (synchronized)
        """
        resource_filter = resource_filter or {}
        kwargs["limit"] = kwargs.get("limit", 100)
        resources = self.search(resource_type, resource_filter, **kwargs)

        return [self.get_resource_by_id(r.id) for r in resources]

    def download_resource(self, resource, path):
        """Download a resource.

        Args:
            resource (kgforge.core.Resource): downloadable resource
            path (str): path to the directory into which the data is downloaded
        """
        file_path = Path(path, resource.name)
        if file_path.is_file():
            L.warning("File %s already exists, not downloading...", file_path)
            return
        if resource.type == "DataDownload" and hasattr(resource, "contentUrl"):
            self._forge.download(resource, "contentUrl", path)
        else:
            raise RuntimeError(f"resource {resource.type} can not be downloaded.")
