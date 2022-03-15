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

# Simplified regexp, see https://www.w3.org/TR/sparql11-query/#QSynIRI for full reference.
_IRI_RE = re.compile(r'^<([^<>"{}|^`\\\x00-\x20])*>$')

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
_NEXUS_KEYS_RE = re.compile(f"(^)?({'|'.join(_NEXUS_KEYS)})")
PREFIX_LIST = """
   PREFIX bmc: <https://bbp.epfl.ch/ontologies/core/bmc/>
   PREFIX bmo: <https://bbp.epfl.ch/ontologies/core/bmo/>
   PREFIX commonshapes: <https://neuroshapes.org/commons/>
   PREFIX datashapes: <https://neuroshapes.org/dash/>
   PREFIX dc: <http://purl.org/dc/elements/1.1/>
   PREFIX dcat: <http://www.w3.org/ns/dcat#>
   PREFIX dcterms: <http://purl.org/dc/terms/>
   PREFIX mba: <http://api.brain-map.org/api/v2/data/Structure/>
   PREFIX nsg: <https://neuroshapes.org/>
   PREFIX nxv: <https://bluebrain.github.io/nexus/vocabulary/>
   PREFIX oa: <http://www.w3.org/ns/oa#>
   PREFIX owl: <http://www.w3.org/2002/07/owl#>
   PREFIX prov: <http://www.w3.org/ns/prov#>
   PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
   PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
   PREFIX schema: <http://schema.org/>
   PREFIX sh: <http://www.w3.org/ns/shacl#>
   PREFIX shsh: <http://www.w3.org/ns/shacl-shacl#>
   PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
   PREFIX vann: <http://purl.org/vocab/vnn/>
   PREFIX void: <http://rdfs.org/ns/void#>
   PREFIX xml: <http://www.w3.org/XML/1998/namespace/>
   PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""

NAMESPACE_MAPPING = {
    "createdBy": USERS_NAMESPACE,
    "updatedBy": USERS_NAMESPACE,
    "project": PROJECTS_NAMESPACE,
}


@contextmanager
def timed(s):
    start_time = time.monotonic()
    try:
        yield
    finally:
        exec_time = time.monotonic() - start_time
        L.info("Execution time for %s: %.3f", s, exec_time)


class NexusConnector:
    def __init__(self, forge, debug=False):
        self._forge = forge
        self._query_builder = QueryBuilder()
        self._search_builder = SearchBuilder()
        self._debug = debug

    def search(
        self, type_, filters, limit=None, offset=None, search_in_graph=False, **kwargs
    ) -> List[Resource]:
        search_filters = self._search_builder.build_filters(type_, filters)

        with timed("search"):
            return self._forge.search(
                search_filters,
                debug=kwargs.get("debug", self._debug),
                limit=limit,
                offset=offset,
                search_in_graph=search_in_graph,
                **kwargs,
            )

    def query(self, query: str, limit=None, offset=None, rewrite=True) -> List[Resource]:
        with timed("query"):
            return self._forge.sparql(
                query, debug=self._debug, limit=limit, offset=offset, rewrite=rewrite
            )

    def get_resource_by_id(self, resource_id: str, version=None, cross_bucket=True) -> Resource:
        with timed("get_resource_by_id"):
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
        # query = self._query_builder.build_query(resource_type, resource_filter)
        # return self.get_resources_by_query(query, limit=limit)
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


class SearchBuilder:
    @staticmethod
    def build_filters(type_, filters):
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


class QueryBuilder:
    _value_transformation = {
        "project": lambda x: f"<{PROJECTS_NAMESPACE}{x}>",
        "createdBy": lambda x: f"<{USERS_NAMESPACE}{x}>",
        "updatedBy": lambda x: f"<{USERS_NAMESPACE}{x}>",
    }

    def _escape(self, value):
        # FIXME: do proper escaping
        if isinstance(value, str) and not _IRI_RE.match(value):
            # v is a string literal
            return repr(value)
        return str(value)

    def _process_item(self, n, key, value, statements, filters):
        split_key = key.split(".")
        key = _NEXUS_KEYS_RE.sub(r"\1nxv:\2", " / ".join(split_key))
        transform = self._value_transformation.get(split_key[-1], lambda x: x)
        if isinstance(value, (list, tuple, set)):
            values = ", ".join(self._escape(transform(v)) for v in value)
            statements.append(f"{key} ?o{n}")
            filters.append(f"FILTER(?o{n} IN ({values}))")
        else:
            value = self._escape(transform(value))
            statements.append(f"{key} {value}")

    def build_query(self, resource_type, resource_filter):
        # see https://www.w3.org/TR/sparql11-query/#pp-language
        statements = [
            f"a {resource_type}",
            "nxv:project ?project",
            "nxv:createdAt ?createdAt",
            "nxv:deprecated false",
        ]
        filters = []
        if isinstance(resource_filter, dict):
            resource_filter = resource_filter.items()
        for n, (key, value) in enumerate(resource_filter):
            self._process_item(n, key, value, statements, filters)

        statements = " ;\n".join(statements)
        filters = " .\n".join(filters)
        query = f"""
            {PREFIX_LIST}
            SELECT ?id ?project
            WHERE {{
                ?id\n{statements} .
                {filters}
            }}
            """
        # ORDER BY DESC(?createdAt)
        # """
        return query
