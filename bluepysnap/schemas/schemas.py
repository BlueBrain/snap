import re
from pathlib import Path

import h5py
import jsonschema
import pkg_resources
import yaml

DEFINITIONS = "definitions"


def _load_schema_file(*args):
    """Load one of the predefined YAML schema files."""
    filename = str(Path(*args).with_suffix(".yaml"))
    with open(pkg_resources.resource_filename(__name__, filename)) as fd:
        return yaml.safe_load(fd)


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
    from bluepysnap.circuit_validation import Error

    warnings = []
    errors = []

    for e in schema_errors:
        if e.path[-1] == "datatype":
            path = join_str.join(list(e.path)[:-1])
            message = f"Incorrect datatype '{e.instance}' for '{path}': {e.message}"
            warnings.append(message)
        else:
            path = []
            for item in e.path:
                # show list index in the [0] format
                if isinstance(item, int):
                    path[-1] += f"[{item}]"
                else:
                    path.append(item)

            # Add special message in case of attribute missing
            if path[-1] == "attributes":
                path = join_str.join(path[:-1])
                message = f"{path}: {e.message} (attribute)"
            else:
                path = join_str.join(path)
                message = f"{path}: {e.message}"
            errors.append(message)

    ret_errors = []

    if len(warnings) > 0:
        message = filepath + ":\n\t" + "\n\t".join(warnings)
        ret_errors.append(Error(Error.WARNING, message))
    if len(errors) > 0:
        message = filepath + ":\n\t" + "\n\t".join(errors)
        ret_errors.append(Error(Error.FATAL, message))

    return ret_errors


def _validate_schema_for_dict(schema, dict_):
    """Run a schema validation for a dictionary."""
    validator = jsonschema.validators.Draft202012Validator(schema)

    return validator.iter_errors(dict_)


def _parse_schema(object_type, sub_type=None):
    """Parses the schema from partial schemas for given object type.

    Args:
        object_type (str): Type of the object. Accepts "edge", "node" and "circuit".
        sub_type (str): Sub type of object. E.g., "biophysical".

    Returns:
        dict: Schema parsed as a dictionary.
    """
    if object_type not in ("edge", "node", "circuit"):
        raise RuntimeError(f"Unknown type: {object_type}")  # refine

    if object_type == "circuit":
        return _load_schema_file(object_type)

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
        h5 (h5.File instance): h5 file to translate

    Returns:
        dict: dictionary of the file structure
    """
    properties = {}

    def get_dataset_dtype(item):
        if item.dtype.hasobject:
            return h5py.check_string_dtype(item.dtype).encoding

        return item.dtype.name

    for key, value in h5.items():
        if isinstance(value, h5py.Group):
            properties[key] = _get_h5_structure_as_dict(value)
        elif isinstance(value, h5py.Dataset):
            properties[key] = {"datatype": get_dataset_dtype(value)}
        else:
            properties[key] = {}

        attrs = dict(value.attrs.items())
        if attrs:
            properties[key]["attributes"] = attrs

    return properties


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

    Returns:
        list: List of errors, empty if no errors
    """
    if virtual:
        edges_type += "_virtual"

    with h5py.File(path) as h5:
        edges_h5_dict = _get_h5_structure_as_dict(h5)

    errors = _validate_schema_for_dict(_parse_schema("edge", edges_type), edges_h5_dict)

    return _wrap_errors(path, errors, "/")