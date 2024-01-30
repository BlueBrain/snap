"""Functions for schema based validation of circuit files."""

from pathlib import Path

import h5py
import importlib_resources
import jsonschema
import numpy as np
import yaml

from bluepysnap.exceptions import BluepySnapValidationError

DEFINITIONS = "definitions"


def _load_schema_file(*args):
    """Load one of the predefined YAML schema files."""
    filename = str(Path(*args).with_suffix(".yaml"))
    filepath = importlib_resources.files(__package__) / filename
    if not filepath.is_file():
        raise FileNotFoundError(f"Schema file {filepath} not found")
    return yaml.safe_load(filepath.read_text())


def _parse_path(path, join_str):
    error_path = []

    for item in path:
        # show list index in the [0] format
        if isinstance(item, int):
            error_path[-1] += f"[{item}]"
        else:
            error_path.append(item)

    return join_str.join(error_path)


def _wrap_errors(filepath, schema_errors, join_str):
    """Handles parsing of schema errors into more meaningful messages.

    Also wraps all the warngings and errors to single Error instances.

    join_str allows showing circuit config paths as:
        path.to.the.error[1].cause
    and h5 dataset paths as:
        path/to/the/error/cause
    """
    # NOTE: would probably make more sense to have different parser for circuit,
    # as datatype and attributes only exists in the case of h5 files.

    warnings = []
    errors = []

    for e in schema_errors:
        if not e.path:
            errors.append(e.message)
        elif e.path[-1] == "datatype":
            path = _parse_path(list(e.path)[:-1], join_str)
            warnings.append(f"incorrect datatype '{e.instance}' for '{path}': {e.message}")
        else:
            if e.schema_path[-1] in e.schema.get("messages", {}):
                path = _parse_path(e.path, join_str)
                message = e.schema["messages"][e.schema_path[-1]]
                message = f"{path}: {message}"
            elif e.schema_path[-1] in ("maxProperties", "minProperties"):
                path = _parse_path(e.path, join_str)
                many_or_few = "many" if e.schema_path[-1] == "maxProperties" else "few"
                message = f"{path}: too {many_or_few} properties"
            elif e.path[-1] == "attributes":
                # Add special message in case of attribute missing
                path = _parse_path(list(e.path)[:-1], join_str)
                attributes = e.schema["required"]
                message = f"{path}: missing required attribute(s) {attributes}"
            elif "attributes" in e.schema.get("required", []):
                path = _parse_path(e.path, join_str)
                attributes = e.schema["properties"]["attributes"]["required"]
                message = f"{path}: missing required attribute(s) {attributes}"
            else:
                path = _parse_path(e.path, join_str)
                message = f"{path}: {e.message}"
            if message not in errors:
                errors.append(message)

    ret_errors = []

    if len(warnings) > 0:
        message = filepath + ":\n\t" + "\n\t".join(warnings)
        ret_errors.append(BluepySnapValidationError.warning(message))
    if len(errors) > 0:
        message = filepath + ":\n\t" + "\n\t".join(errors)
        ret_errors.append(BluepySnapValidationError.fatal(message))

    return ret_errors


def _validate_schema_for_dict(schema, dict_):
    """Run a schema validation for a dictionary."""
    validator = jsonschema.validators.Draft202012Validator(schema)

    return validator.iter_errors(dict_)


def _parse_schema(object_type, sub_type=None):
    """Parses the schema from partial schemas for given object type.

    Args:
        object_type (str): Type of the object. Accepts "edge", "node", "circuit" and "simulation".
        sub_type (str): Sub type of object. E.g., "biophysical".

    Returns:
        dict: Schema parsed as a dictionary.
    """
    if object_type == "circuit":
        return _load_schema_file(object_type)
    elif object_type == "simulation":
        schema = _load_schema_file(DEFINITIONS, "simulation_input")
        schema.update(_load_schema_file(object_type))
        return schema
    elif object_type not in ("edge", "node"):
        raise RuntimeError(f"Unknown object type: {object_type}")

    schema = _load_schema_file(DEFINITIONS, "datatypes")
    schema.update(_load_schema_file(DEFINITIONS, object_type))
    schema.update(_load_schema_file(object_type, sub_type))

    return schema


def _get_h5_structure_as_dict(h5):
    """Recursively translates h5 file into a dictionary.

    For groups, the subgroups/datasets are translated as a subdictionary.
    Datatype of datasets is resolved and returned as {'datatype': <dtype>}.
    Attributes of either groups or datasets are returned as {'attributes': {<key>: <value>}}.

    Args:
        h5 (h5.File, h5.Group): h5 file or group to translate

    Returns:
        dict: dictionary of the structure
    """
    properties = {}

    def get_dataset_dtype(item):
        if item.dtype.hasobject:
            return h5py.check_string_dtype(item.dtype).encoding

        return item.dtype.name

    for key, value in h5.items():
        if isinstance(value, h5py.Group):
            properties[key] = _get_h5_structure_as_dict(value)
        else:  # Dataset
            properties[key] = {"datatype": get_dataset_dtype(value)}

        attrs = dict(value.attrs.items())
        if attrs:
            properties[key]["attributes"] = attrs

    # Resolve enumerated properties' types to check their actual data type
    if "@library" in properties:
        for key, value in properties["@library"].items():
            properties[key] = value

    return properties


def validate_simulation_schema(path, config):
    """Validates a simulation config against a schema.

    Args:
        path (str): path to the config (for error messages)
        config (dict): resolved bluepysnap config

    Returns:
        list: List of errors, empty if no errors
    """
    errors = _validate_schema_for_dict(_parse_schema("simulation"), config)

    return _wrap_errors(path, errors, ".")


def validate_circuit_schema(path, config):
    """Validates a circuit config against a schema.

    Args:
        path (str): path to the config (for error messages)
        config (dict): resolved bluepysnap config

    Returns:
        list: List of errors, empty if no errors
    """
    errors = _validate_schema_for_dict(_parse_schema("circuit"), config)

    return _wrap_errors(path, errors, ".")


def validate_nodes_schema(path, nodes_type):
    """Validates a nodes file against a schema.

    Args:
        path (str): path to the nodes file
        nodes_type (str): node type (e.g., "biophysical")

    Returns:
        list: List of errors, empty if no errors
    """
    with h5py.File(path) as h5:
        nodes_h5_dict = _get_h5_structure_as_dict(h5)

    errors = _validate_schema_for_dict(_parse_schema("node", nodes_type), nodes_h5_dict)

    return _wrap_errors(path, errors, "/")


def validate_edges_schema(path, edges_type, virtual):
    """Validates an edges file against a schema.

    Args:
        path (str): path to the edges file
        edges_type (str): edge type (e.g., "chemical")
        virtual(bool): whether this is a virtual edge population

    Returns:
        list: List of errors, empty if no errors
    """
    if virtual:
        edges_type += "_virtual"

    with h5py.File(path) as h5:
        edges_h5_dict = _get_h5_structure_as_dict(h5)

    errors = _validate_schema_for_dict(_parse_schema("edge", edges_type), edges_h5_dict)

    return _wrap_errors(path, errors, "/")


def _resolve_types(resolver, types):
    """Use jsonschema `resolver` to resolve the `types` dict."""
    cache = {}

    def _resolve_type(type_):
        if type_ not in cache:
            type_ = resolver.resolve(type_)[1]["properties"]["datatype"]["const"]
            if hasattr(np, type_):
                cache[type_] = getattr(np, type_)
            elif type_ == "utf-8":
                cache[type_] = str

        return cache[type_]

    return {k: _resolve_type(v["$ref"]) for k, v in types.items()}


def _get_reference_resolver(schema):
    """Get reference resolver for the given schema."""
    return jsonschema.validators.RefResolver("", schema)


def nodes_schema_types(nodes_type):
    """Get the datatypes of the attributes for nodes.

    Args:
        nodes_type (str): node type (e.g., "biophysical")

    Returns:
        dict: name -> type of column
    """
    schema = _parse_schema("node", nodes_type)
    resolver = _get_reference_resolver(schema)

    schema = schema["$node_file_defs"]["nodes_file_root"]["properties"]["nodes"]
    schema = schema["patternProperties"][""]["properties"]["0"]["properties"]
    dynamics_params = schema["dynamics_params"]["properties"]
    del schema["dynamics_params"]
    del schema["@library"]

    return _resolve_types(resolver, schema), _resolve_types(resolver, dynamics_params)


def edges_schema_types(edges_type, virtual):
    """Get the datatypes of the attributes for edges.

    Args:
        edges_type (str): edges type (e.g., "chemical")
        virtual(bool): whether this is a virtual edge population

    Returns:
        dict: name -> type of column
    """
    if virtual:
        edges_type += "_virtual"

    schema = _parse_schema("edge", edges_type)
    resolver = _get_reference_resolver(schema)

    schema = schema["$edge_file_defs"]["edges_file_root"]["properties"]["edges"]
    schema = schema["patternProperties"][""]["properties"]["0"]["properties"]
    del schema["@library"]
    del schema["synapse_id"]

    return _resolve_types(resolver, schema)
