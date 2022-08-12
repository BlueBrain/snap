from pathlib import Path

import pkg_resources
import yaml

SCHEMAS = {
    "definitions": {
        "edge": "edge_file_definitions.yaml",
        "node": "node_file_definitions.yaml",
        "type": "types.yaml",
    },
    "edge": {
        "chemical": "edges_chemical.yaml",
    },
    "node": {"biophysical": "nodes_biophysical.yaml", "virtual": "nodes_virtual.yaml"},
    "circuit": "circuit_config.yaml",
}


def _load_schema_file(filename):
    """Load one of the predefined YAML schema files."""
    with open(pkg_resources.resource_filename(__name__, filename)) as fd:
        return yaml.safe_load(fd)


def parse_schema(object_type, sub_type=None, virtual=False):
    """Parses the schema from partial schemas for given object type.

    Args:
        object_type (str): Type of the object. Accepts "edge", "node" and "circuit".
        sub_type (str): Sub type of object. E.g., "biophysical".
        virtual (bool): A flag indicating if the object is virtual.

    Returns:
        dict: Schema parsed as a dictionary.
    """
    if object_type not in ("edge", "node", "circuit"):
        raise RuntimeError(f"Unknown type: {object_type}")  # refine

    if object_type == "circuit":
        return _load_schema_file(object_type)

    schema = _load_schema_file(SCHEMAS["definitions"]["type"])
    schema.update(_load_schema_file(SCHEMAS["definitions"][object_type]))

    if sub_type == "chemical" and virtual:
        sub_type = "chemical_virtual"

    schema.update(_load_schema_file(SCHEMAS[object_type][sub_type]))
    return schema
