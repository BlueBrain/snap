import logging
import re
import time
from contextlib import contextmanager
from typing import List

from kgforge.core import Resource

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
    "rev",
    "self",
    "updatedAt",
    "updatedBy",
}
_NEXUS_KEYS_RE = re.compile(f"(^)?({'|'.join(_NEXUS_KEYS)})")


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
        self._debug = debug

    def query(self, query: str, limit=None, offset=None) -> List[Resource]:
        with timed("query"):
            return self._forge.sparql(query, debug=self._debug, limit=limit, offset=offset)

    def get_resource_by_id(self, resource_id: str, version=None, cross_bucket=True) -> Resource:
        with timed("get_resource_by_id"):
            return self._forge.retrieve(resource_id, version=version, cross_bucket=cross_bucket)

    def get_resources_by_query(self, query: str, limit=None, offset=None) -> List[Resource]:
        result = self.query(query, limit=limit, offset=offset)
        # TODO: execute requests concurrently, or pass a list of ids if possible,
        #  to avoid calling the nexus endpoint for each resource.
        return [self.get_resource_by_id(r.id) for r in result]

    def get_resources(self, resource_type: str, resource_filter=(), limit=100) -> List[Resource]:
        query = self._query_builder.build_query(resource_type, resource_filter)
        return self.get_resources_by_query(query, limit=limit)


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
        key = _NEXUS_KEYS_RE.sub(r"\1nexus:\2", " / ".join(split_key))
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
            "nexus:project ?project",
            "nexus:createdAt ?createdAt",
            "nexus:deprecated false",
        ]
        filters = []
        if isinstance(resource_filter, dict):
            resource_filter = resource_filter.items()
        for n, (key, value) in enumerate(resource_filter):
            self._process_item(n, key, value, statements, filters)

        statements = ";\n".join(statements)
        filters = ".\n".join(filters)
        query = f"""
            PREFIX nexus: <{NEXUS_NAMESPACE}>
            SELECT ?id ?project
            WHERE {{
                ?id\n{statements} .
                {filters}
            }}
            ORDER BY DESC(?createdAt)
            """
        return query
