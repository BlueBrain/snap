import logging
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List

from kgforge.core import Resource
from more_itertools import always_iterable

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
    def __init__(self, forge, debug=False):
        self._forge = forge
        self._debug = debug

    def search(
        self, type_, filters, limit=None, offset=None, search_in_graph=False, **kwargs
    ) -> List[Resource]:
        search_filters = build_search_filters(type_, filters)

        return self._forge.search(
            search_filters,
            debug=kwargs.get("debug", self._debug),
            limit=limit,
            offset=offset,
            search_in_graph=search_in_graph,
            **kwargs,
        )

    def query(self, query: str, limit=None, offset=None, rewrite=True) -> List[Resource]:
        return self._forge.sparql(
            query, debug=self._debug, limit=limit, offset=offset, rewrite=rewrite
        )

    def get_resource_by_id(self, resource_id: str, version=None, cross_bucket=True) -> Resource:
        return self._forge.retrieve(resource_id, version=version, cross_bucket=cross_bucket)

    def get_resources_by_query(
        self, query: str, limit=None, offset=None, rewrite=True
    ) -> List[Resource]:
        result = self.query(query, limit=limit, offset=offset, rewrite=rewrite)
        # TODO: execute requests concurrently, or pass a list of ids if possible,
        #  to avoid calling the nexus endpoint for each resource.
        return [self.get_resource_by_id(r.id) for r in result]

    def get_resources(
        self, resource_type: str, resource_filter=None, limit=100, **kwargs
    ) -> List[Resource]:
        resource_filter = resource_filter or {}
        resources = self.search(resource_type, resource_filter, limit, **kwargs)
        return [self.get_resource_by_id(r.id) for r in resources]

    def download_resource(self, resource, path):
        file_path = Path(path, resource.name)
        if file_path.is_file():
            L.warning("File %s already exists, not downloading...", file_path)
            return
        if resource.type == "DataDownload" and hasattr(resource, "contentUrl"):
            self._forge.download(resource, "contentUrl", path)
        else:
            raise RuntimeError(f"resource {resource.type} can not be downloaded.")
