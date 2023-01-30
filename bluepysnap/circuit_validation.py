"""Standalone module that validates Sonata circuit. See ``validate`` function.

The idea here is to not depend on libsonata if possible, so we can use this in all situations
"""
import logging
from pathlib import Path

import click
import h5py
import numpy as np
import pandas as pd

from bluepysnap import schemas
from bluepysnap.config import Parser
from bluepysnap.morph import EXTENSIONS_MAPPING
from bluepysnap.sonata_constants import DEFAULT_EDGE_TYPE, DEFAULT_NODE_TYPE
from bluepysnap.utils import load_json

L = logging.getLogger("brainbuilder")
MAX_MISSING_FILES_DISPLAY = 10


class Error:
    """Error used for reporting of validation errors."""

    FATAL = "FATAL"
    WARNING = "WARNING"
    INFO = "INFO"

    def __init__(self, level, message=None):
        """Error.

        Args:
            level (str): error level
            message (str|None): message
        """
        self.level = level
        self.message = message

    def __str__(self):
        """Returns only message by default."""
        return str(self.message)

    __repr__ = __str__

    def __eq__(self, other):
        """Two errors are equal if inherit from Error and their level, message are equal."""
        if not isinstance(other, Error):
            return False
        return self.level == other.level and self.message == other.message

    def __hash__(self):
        """Hash. Errors with the same level and message give the same hash."""
        return hash(self.level) ^ hash(self.message)


def fatal(message):
    """Shortcut for a fatal error.

    Args:
        message (str): text message

    Returns:
        Error: Error with level FATAL
    """
    return Error(Error.FATAL, message)


def _check_components_dir(name, components):
    """Checks existence of directory within Sonata config 'components'.

    Args:
        name (str): components directory name
        components (dict): ref to config's components

    Returns:
        list: List of errors, empty if no errors
    """
    dirpath = components.get(name)
    if not dirpath or not Path(dirpath).is_dir():
        return [fatal(f'Invalid components "{name}": {dirpath}')]
    return []


def _check_files(name, files, level):
    """Checks for existence of files within an h5 group.

    Args:
        name (str): h5 group name
        files (Iterable): files to check for existence
        level (str): level of generated errors for missing files

    Returns:
        list: List of errors, empty if no errors
    """
    files = set(files)
    missing = []
    for f in sorted(files):
        if not f.is_file():
            missing.append(f)

        if len(missing) >= MAX_MISSING_FILES_DISPLAY:
            break

    if missing:
        filenames = "".join(f"\t{m.name}\n" for m in missing)

        return [
            Error(level, f"missing at least {len(missing)} files in group {name}:\n{filenames}")
        ]

    return []


def _print_errors(errors):
    """Some fancy errors printing."""
    colors = {Error.WARNING: "yellow", Error.FATAL: "red", Error.INFO: "green"}

    if not errors:
        print(click.style("No Error: Success.", fg=colors[Error.INFO]))

    for error in errors:
        print(click.style(error.level + ": ", fg=colors[error.level]) + str(error))


def _check_duplicate_populations(networks, key):
    """Check that that for key = nodes|edges, no duplicate populations names exists."""
    seen = set()
    errors = []
    for network in networks.get(key, {}):
        for population in network.get("populations", {}):
            if population in seen:
                errors.append(
                    fatal(f'Already have population "{population}" in config for type "{key}"')
                )
            seen.add(population)

    return errors


def _find_nodes_population(node_population_name, nodes):
    """Finds node population item in config.

    Args:
        node_population_name (str): name of node population
        nodes (list): "nodes" part of the resolved bluepysnap config

    Returns:
        dict/None: A dict item of "nodes" of the resolved bluepysnap config.
        None if it finds nothing.
    """
    for nodes_dict in nodes:
        if node_population_name in nodes_dict.get("populations", {}):
            return nodes_dict

    return None


def _get_group_name(group, parents=0):
    """Gets group name of h5 group.

    Args:
        group (h5py.Group): nodes group in nodes .h5 file
        parents (int): number of extra parents needed
    """
    return Path(*Path(group.name).parts[-(parents + 1) :])


def _get_model_template_file(model_template):
    """Resolves 'model_template' field of nodes group to a proper filename."""
    parts = model_template.split(":", 1)
    return parts[1] + "." + parts[0]


def _nodes_group_to_dataframe(group, population):
    """Transforms hdf5 population group to pandas DataFrame.

    Args:
        group: HDF5 nodes group
        population: HDF5 nodes population

    Returns:
        pd.DataFrame: dataframe with all group attributes
    """
    # TODO: remove multi-indexing (BBP only supports group '0')
    df = pd.DataFrame(population["node_type_id"][:], columns=["type_id"])
    size = df.size
    df["id"] = population["node_id"] if "node_id" in population else np.arange(size)
    df["group_id"] = population["node_group_id"] if "node_group_id" in population else 0
    df["group_index"] = (
        population["node_group_index"] if "node_group_index" in population else np.arange(size)
    )
    df = df[df["group_id"] == int(str(_get_group_name(group)))]

    for k, v in group.items():
        if k == "@library":
            continue
        if isinstance(v, h5py.Dataset):
            if v.dtype == h5py.string_dtype():
                df[k] = v.asstr()[:]
            else:
                df[k] = v[:]

    if "@library" in group:
        for k, v in group["@library"].items():
            if isinstance(v, h5py.Dataset):
                df[k] = pd.Categorical.from_codes(df[k], categories=v.asstr()[:])

    return df


def _check_bio_nodes_group(group_df, group, population, population_name):
    """Checks biophysical nodes group for errors.

    Args:
        group_df (pd.DataFrame): nodes group as a dataframe
        group (h5py.Group): nodes group in nodes .h5 file
        population (dict): a merged dictionary (current population and 'components' in config)
        population_name (str): name of the population

    Returns:
        list: List of errors, empty if no errors
    """
    L.debug("Check biophysical nodes group")

    errors = []

    group_name = _get_group_name(group, parents=1)

    morph_dirs = set()
    if "morphologies_dir" in population:
        dir_errors = _check_components_dir("morphologies_dir", population)
        errors += dir_errors
        if len(dir_errors) == 0:
            morph_dirs = {(population["morphologies_dir"], "swc")}

    if "alternate_morphologies" in population:
        for morph_type, morph_path in population["alternate_morphologies"].items():
            dir_errors = _check_components_dir(morph_type, population["alternate_morphologies"])
            errors += dir_errors
            if len(dir_errors) == 0:
                for extension, _type in EXTENSIONS_MAPPING.items():
                    if _type == morph_type:
                        morph_dirs |= {(morph_path, extension)}

    if "morphologies_dir" not in population and "alternate_morphologies" not in population:
        errors.append(
            fatal(
                "at least one of 'morphologies_dir' or 'alternate_morphologies' "
                f"must to be defined for 'biophysical' population '{population_name}'"
            )
        )

    if "morphology" in group_df.columns:
        for morph_path, extension in morph_dirs:
            L.debug("Checking morph files (%s): %s", extension, morph_path)

            errors += _check_files(
                f"morphology: {group_name}[{group.file.filename}]",
                (Path(morph_path, m + "." + extension) for m in group_df["morphology"].unique()),
                Error.WARNING,
            )

    if "biophysical_neuron_models_dir" in population:
        errors += _check_components_dir("biophysical_neuron_models_dir", population)

        bio_path = Path(population["biophysical_neuron_models_dir"])
        L.debug("Checking neuron model files: %s", bio_path)
        errors += _check_files(
            f"model_template: {group_name}[{group.file.filename}]",
            (
                bio_path / _get_model_template_file(m)
                for m in group_df.get("model_template", pd.Series(dtype="object")).unique()
            ),
            Error.WARNING,
        )
    else:
        errors.append(
            fatal(f"'biophysical_neuron_models_dir' not defined for population '{population_name}'")
        )
    return errors


def _check_nodes_group(group_df, group, population, population_name):
    """Validates nodes group in nodes population.

    Args:
        group_df (pd.DataFrame): nodes group in nodes .h5 file
        group (h5py.Group): nodes group in nodes .h5 file
        population (dict): the node population config merged with the components config
        population_name (str): the name of the population

    Returns:
        list: List of errors, empty if no errors
    """
    L.debug("Check nodes group: %s", group.name)

    errors = []
    if "model_type" in group_df and group_df["model_type"][0] != population["type"]:
        message = (
            f"Population '{population_name}' type mismatch: "
            f"'{group_df['model_type'][0]}' (nodes_file), "
            f"'{population['type']}' (config)"
        )
        errors.append(Error(Error.WARNING, message))

    if population["type"] == "biophysical":
        return errors + _check_bio_nodes_group(group_df, group, population, population_name)

    return errors


def validate_node_population(nodes_file, population_dict, name):
    """Validates nodes population.

    Args:
        nodes_file (str): path to the nodes file (.h5)
        population_dict (dict): the node population config merged with the components config
        config (dict): resolved bluepysnap config

    Returns:
        list: List of errors, empty if no errors
    """
    with h5py.File(nodes_file, "r") as h5f:
        if "nodes" not in h5f or len(h5f["nodes"]) == 0:
            return []

        # special case in which there are populations but not the one expected
        if name not in h5f["nodes"]:
            return [fatal(f"population '{name}' not found in {nodes_file}")]

        population = h5f[f"nodes/{name}"]

        if "0" in population and "node_type_id" in population:
            group = population["0"]
            group_df = _nodes_group_to_dataframe(group, population)
            if len(group_df) > 0:
                return _check_nodes_group(group_df, group, population_dict, name)

    return []


def _get_node_ids(nodes_h5, population_name):
    """Gets node ids of node population.

    Args:
        nodes_h5: (h5py.File): nodes file h5 instance
        population_name (str): node population name

    Returns:
        np.ndarray: Numpy array of node ids, empty if couldn't find any
    """
    if f"nodes/{population_name}" in nodes_h5:
        node_population = nodes_h5["nodes"][population_name]
        if "node_id" in node_population:
            return node_population["node_id"][:]
        elif "0" in node_population:
            for attr in node_population["0"].values():
                if isinstance(attr, h5py.Dataset):
                    return np.arange(len(attr))

    return np.empty(0)


def _check_edges_node_ids(nodes_ds, nodes):
    """Checks that nodes ids in edges can be resolved to nodes ids in nodes populations.

    Args:
        nodes_ds (h5py.Dataset): nodes dataset in edges population
        nodes (list): "nodes" part of the resolved bluepysnap config

    Returns:
        list: List of errors, empty if no errors
    """
    L.debug("Check edges node ids: %s", nodes_ds.name)

    if "node_population" not in nodes_ds.attrs:
        return []

    node_population_name = nodes_ds.attrs["node_population"]

    nodes_dict = _find_nodes_population(node_population_name, nodes)
    if not nodes_dict:
        return [fatal(f'No node population for "{nodes_ds.name}"')]

    if "nodes_file" not in nodes_dict or not Path(nodes_dict["nodes_file"]).is_file():
        return []

    errors = []
    with h5py.File(nodes_dict["nodes_file"], "r") as h5f:
        node_ids = _get_node_ids(h5f, node_population_name)
        if node_ids.size > 0:
            missing_ids = sorted(set(nodes_ds[:]) - set(node_ids))
            if missing_ids:
                errors.append(
                    fatal(f"{nodes_ds.name} misses node ids in its node population: {missing_ids}")
                )
        elif f"nodes/{node_population_name}" in h5f:
            errors.append(fatal((f"{nodes_ds.name} does not have node ids in its node population")))

    return errors


def _check_edges_indices(population):
    """Check edges population indices.

    Args:
        population (h5py.Group): edges population

    Returns:
        list: List of errors, empty if no errors
    """
    L.debug("Check edges indices: %s", population.name)

    def _check(indices, nodes_ds):
        """The main indices check.

        It iterates over edge indices and verifies that each has its
        nodes in place in nodes populations
        """
        nodes_ranges = indices["node_id_to_ranges"]
        node_to_edges_ranges = indices["range_to_edge_id"]
        for node_id, nodes_range in enumerate(nodes_ranges[:]):
            if 0 <= nodes_range[0] < nodes_range[1]:
                edges_range = node_to_edges_ranges[nodes_range[0] : nodes_range[1]][0]
                edge_node_ids = list(set(nodes_ds[edges_range[0] : edges_range[1]]))
                if len(edge_node_ids) > 1 or edge_node_ids[0] != node_id:
                    errors.append(
                        fatal(
                            f"Population {population.file.filename} edges {edge_node_ids} have "
                            f"node ids {edges_range} instead of single id {node_id}"
                        )
                    )

    errors = []
    indices = population["indices"]
    source_to_target = indices["source_to_target"] if "source_to_target" in indices else None
    target_to_source = indices["target_to_source"] if "target_to_source" in indices else None

    # These are "optional" (not mentioned in our spec) but better to at least give a warning
    if not source_to_target:
        errors.append(Error(Error.WARNING, f'No "source_to_target" in {population.file.filename}'))

    if not target_to_source:
        errors.append(Error(Error.WARNING, f'No "target_to_source" in {population.file.filename}'))

    if target_to_source and source_to_target:
        if "source_node_id" in population:
            _check(source_to_target, population["source_node_id"])
        if "target_node_id" in population:
            _check(target_to_source, population["target_node_id"])

    return errors


def _check_edge_population_data(population, nodes):
    """Check edges population data.

    Args:
        population (h5py.Group): edges population
        nodes (list): "nodes" part of the resolved bluepysnap config

    Returns:
        list: List of errors, empty if no errors
    """
    L.debug("Check edges population data")

    errors = []

    if "source_node_id" in population:
        errors += _check_edges_node_ids(population["source_node_id"], nodes)

    if "target_node_id" in population:
        errors += _check_edges_node_ids(population["target_node_id"], nodes)

    if "indices" in population:
        errors += _check_edges_indices(population)
    else:  # "optional" (not mentioned in our spec) but better to at least give a warning
        errors.append(Error(Error.WARNING, f'No "indices" in {population.file.filename}'))

    return errors


def validate_edge_population(edges_file, name, nodes):
    """Validate an edge population.

    Args:
        edges_file (str): path to the edges file
        name (str): name of the population
        nodes (list): "nodes" listing of the config

    Returns:
        list: List of errors, empty if no errors
    """
    with h5py.File(edges_file, "r") as h5f:
        if "edges" not in h5f or len(h5f["edges"]) == 0:
            return []

        # special case in which there are populations but not the one expected
        if name not in h5f["edges"]:
            return [fatal(f"population '{name}' not found in {edges_file}")]

        population = h5f[f"edges/{name}"]

        if "0" in population:
            return _check_edge_population_data(population, nodes)

    return []


def validate_edges_dict(edges_dict, nodes, skip_slow):
    """Validate an item in the "edges" list.

    Args:
        edges_dict (dict): edges population, represented by an item of "edges" in ``config``
        nodes (list): "nodes" part of the resolved bluepysnap config
        skip_slow(bool): skip slow tests

    Returns:
        list: List of errors, empty if no errors
    """
    errors = []

    def _is_source_node_virtual(edges_dict, edge_population, nodes):
        """Check if source node is virtual.

        The required attributes are different for edges with virtual source nodes.
        """
        with h5py.File(edges_dict["edges_file"], "r") as h5:
            source = h5.get(f"edges/{edge_population}/source_node_id")
            source_population = source.attrs.get("node_population") if source else None

        if source_population:
            nodes_dict = _find_nodes_population(source_population, nodes)
            if nodes_dict is not None:
                return nodes_dict["populations"][source_population].get("type") == "virtual"

        return False

    for name, population in edges_dict.get("populations", {}).items():
        pop_type = population.get("type", DEFAULT_EDGE_TYPE)
        edges_file = edges_dict["edges_file"]

        if Path(edges_file).is_file():
            virtual = False
            if pop_type == "chemical":
                virtual = _is_source_node_virtual(edges_dict, name, nodes)
            errors += schemas.validate_edges_schema(edges_file, pop_type, virtual)
            if not skip_slow:
                errors += validate_edge_population(edges_file, name, nodes)
        else:
            errors.append(fatal(f'Invalid "edges_file": {edges_file}'))

    return errors


def validate_nodes_dict(nodes_dict, components):
    """Validate an item in the "nodes" list.

    Args:
        nodes_dict (dict): nodes population, represented by an item of "nodes" in ``config``
        components(dict): "components" part of the ``config``

    Returns:
        list: List of errors, empty if no errors
    """
    errors = []
    for pop_name, pop_dict in nodes_dict.get("populations", {}).items():
        population = {**components, **pop_dict}
        population["type"] = population.get("type", DEFAULT_NODE_TYPE)
        nodes_file = nodes_dict["nodes_file"]

        if Path(nodes_file).is_file():
            errors = schemas.validate_nodes_schema(nodes_file, population["type"])
            errors += validate_node_population(nodes_file, population, pop_name)
        else:
            errors.append(fatal(f'Invalid "nodes_file": {nodes_file}'))

    return errors


def validate_networks(config, skip_slow):
    """Validate "networks" part of the config.

    Acts as a starting point of validation.
    """
    errors = []
    errors += _check_duplicate_populations(config["networks"], "nodes")
    errors += _check_duplicate_populations(config["networks"], "edges")

    components = config.get("components", {})
    nodes = config["networks"].get("nodes", [])

    for nodes_dict in nodes:
        if "nodes_file" in nodes_dict:
            errors += validate_nodes_dict(nodes_dict, components)
    for edges_dict in config["networks"].get("edges", []):
        if "edges_file" in edges_dict:
            errors += validate_edges_dict(edges_dict, nodes, skip_slow)

    return errors


def validate(config_file, skip_slow, only_errors=False, print_errors=True):
    """Validates Sonata circuit.

    Args:
        config_file (str): path to Sonata circuit config file official checks.
        skip_slow (bool): skip slow tests
        only_errors (bool): only return/print fatal errors
        print_errors (bool): print errors

    Returns:
        list: List of errors, empty if no errors
    """
    config = Parser.parse(load_json(config_file), str(Path(config_file).parent))
    errors = schemas.validate_circuit_schema(config_file, config)

    if "networks" in config:
        errors += validate_networks(config, skip_slow)

    if only_errors:
        errors = [e for e in errors if e.level == Error.FATAL]

    if print_errors:
        _print_errors(errors)

    return set(errors)
