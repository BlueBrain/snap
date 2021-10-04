"""Standalone module that validates Sonata circuit. See ``validate`` function."""
import itertools as it
from pathlib import Path

import numpy as np
import pandas as pd
import click
import h5py

from bluepysnap import BluepySnapError
from bluepysnap.config import Config
from bluepysnap.morph import EXTENSIONS_MAPPING
from bluepysnap.bbp import NODE_TYPES, EDGE_TYPES

MAX_MISSING_FILES_DISPLAY = 10


class Error:
    """Error used for reporting of validation errors."""
    FATAL = 'FATAL'
    WARNING = 'WARNING'
    INFO = 'INFO'

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

    def __eq__(self, other):
        """Two errors are equal if inherit from Error and their level, message are equal."""
        if not isinstance(other, Error):
            return False
        return self.level == other.level and self.message == other.message

    def __hash__(self):
        """Hash. Errors with the same level and message give the same hash."""
        return hash(self.level) ^ hash(self.message)


class BbpError(Error):
    """Special class of errors for BBP specification of Sonata."""


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
        return [BbpError(Error.FATAL, f'Invalid components "{name}": {dirpath}')]
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
    missing = sorted({f for f in files if not f.is_file()})
    if missing:
        examples = [e.name for e in it.islice(missing, MAX_MISSING_FILES_DISPLAY)]
        if len(missing) > MAX_MISSING_FILES_DISPLAY:
            examples.append('...')
        filenames = ''.join(f'\t{e}\n' for e in examples)
        return [Error(level, f'missing {len(missing)} files in group {name}:\n{filenames}')]
    return []


def _print_errors(errors):
    """Some fancy errors printing."""
    colors = {
        Error.WARNING: 'yellow',
        Error.FATAL: 'red',
        Error.INFO: 'green'
    }

    if not errors:
        print(click.style('No Error: Success.', fg=colors[Error.INFO]))

    for error in errors:
        print(click.style(error.level + ': ', fg=colors[error.level]) + str(error))


def _check_required_datasets(config):
    """Validates required datasets of "nodes" and "edges" in config.

    Args:
        config (dict): resolved bluepysnap config

    Returns:
        list: List of errors, empty if no errors
    """
    errors = []
    networks = config.get('networks')
    if not networks:
        errors.append(fatal('No "networks" in config'))
        return errors
    nodes = networks.get('nodes')
    if not nodes:
        errors.append(fatal('No "nodes" in config "networks"'))
    edges = networks.get('edges')
    if not edges:
        errors.append(fatal('No "edges" in config "networks"'))
    if not nodes or not edges:
        return errors

    for nodes_dict in nodes:
        nodes_file = nodes_dict.get('nodes_file')
        if nodes_file is None or not Path(nodes_file).is_file():
            errors.append(fatal(f'Invalid "nodes_file": {nodes_file}'))
        types_file = nodes_dict.get('node_types_file')
        if types_file is not None and not Path(types_file).is_file():
            errors.append(fatal(f'Invalid "node_types_file": {types_file}'))
        if not nodes_dict.get('populations'):
            errors.append(BbpError(Error.FATAL, 'No "populations" defined in config "nodes"'))

    for edges_dict in edges:
        edges_file = edges_dict.get('edges_file')
        if edges_file is None or not Path(edges_file).is_file():
            errors.append(fatal(f'Invalid "edges_file": {edges_file}'))
        types_file = edges_dict.get('edge_types_file')
        if types_file is not None and not Path(types_file).is_file():
            errors.append(fatal(f'Invalid "edge_types_file": {types_file}'))
        if not edges_dict.get('populations'):
            errors.append(BbpError(Error.FATAL, 'No "populations" defined in config "edges"'))

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
        nodes_file = nodes_dict.get('nodes_file')
        if nodes_file:
            with h5py.File(nodes_file, 'r') as h5f:
                if '/nodes/' + node_population_name in h5f:
                    return nodes_dict
    return None


def _get_group_name(group, parents=0):
    """Gets group name of h5 group.

    Args:
        group (h5py.Group): nodes group in nodes .h5 file
        parents (int): number of extra parents needed
    """
    return Path(*Path(group.name).parts[-(parents + 1):])


def _get_h5_data(h5, path):
    """Resolves and returns an h5 group/dataset by a path. Returns None if didn't find any.

    Args:
        h5: h5 file or group
        path: path within ``h5``

    Returns:
        h5py.Group/h5py.Dataset/None: A resolved h5 group/dataset or None if it finds nothing.
    """
    return h5[path] if path in h5 else None


def _get_model_template_file(model_template):
    """Resolves 'model_template' field of nodes group to a proper filename."""
    parts = model_template.split(':', 1)
    return parts[1] + '.' + parts[0]


def _get_population_groups(population_h5):
    """Get groups from an edge or node population."""
    return [population_h5[name] for name in population_h5
            if isinstance(population_h5[name], h5py.Group) and name.isdigit()]


def _nodes_group_to_dataframe(group, types_file, population):
    """Transforms hdf5 population group to pandas DataFrame.

    Args:
        group: HDF5 nodes group
        types_file: path to 'node_types_file' of Sonata config
        population: HDF5 nodes population

    Returns:
        pd.DataFrame: dataframe with all group attributes
    """
    df = pd.DataFrame(population['node_type_id'], columns=['type_id'])
    size = df.size
    df['id'] = population['node_id'] if 'node_id' in population else np.arange(size)
    df['group_id'] = population['node_group_id'] if 'node_group_id' in population else 0
    df['group_index'] = population['node_group_index'] \
        if 'node_group_index' in population else np.arange(size)
    df = df[df['group_id'] == int(str(_get_group_name(group)))]
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            if v.dtype == h5py.string_dtype():
                df[k] = v.asstr()[:]
            else:
                df[k] = v[:]
    if '@library' in group:
        for k, v in group['@library'].items():
            if isinstance(v, h5py.Dataset):
                df[k] = v.asstr()[:][df[k].to_numpy(dtype=int)]
    if types_file is None:
        return df
    types = pd.read_csv(types_file, sep=r'\s+')
    # pylint seems to think that types has no member columns
    types.rename(columns={types.columns[0]: 'type_id'}, inplace=True)  # pylint: disable=no-member
    return pd.merge(df, types, on='type_id', how='left')


def _get_group_size(group_h5):
    """Gets size of an edges or nodes group."""
    for name in group_h5:
        if isinstance(group_h5[name], h5py.Dataset):
            return group_h5[name].shape[0]
    raise BluepySnapError(f'Empty group {group_h5}')


def _check_multi_groups(group_id_h5, group_index_h5, population):
    """Checks multiple groups of nodes or edges population."""
    group_id_h5 = group_id_h5[:]
    group_index_h5 = group_index_h5[:]
    if len(group_id_h5) != len(group_index_h5):
        return [fatal(f'Population {population.name} of {population.file.filename} has '
                      'different sizes of "group_id" and "group_index"')]
    group_ids = np.unique(group_id_h5)
    group_names = [_get_group_name(group).name for group in _get_population_groups(population)]
    missing_groups = set(group_ids) - set(np.array(group_names, dtype=int))
    if missing_groups:
        return [fatal(f'Population {population.name} of {population.file.filename} '
                      f'misses group(s): {missing_groups}')]
    for group_id in group_ids:
        group = population[str(group_id)]
        max_id = group_index_h5[group_id_h5 == int(group_id)].max()
        if _get_group_size(group) < max_id:
            return [fatal(f'Group {_get_group_name(group, parents=1)} in file '
                          f'{population.file.filename} should have ids up to {max_id}')]
    return []


def _check_bio_nodes_group(group_df, group, population):
    """Checks biophysical nodes group for errors.

    Args:
        group_df (pd.DataFrame): nodes group as a dataframe
        group (h5py.Group): nodes group in nodes .h5 file
        population (dict): a merged dictionary (current population and 'components' in config)

    Returns:
        list: List of errors, empty if no errors
    """
    def _check_rotations():
        """Checks for proper rotation fields."""
        angle_fields = {'rotation_angle_xaxis', 'rotation_angle_yaxis', 'rotation_angle_zaxis'}
        has_angle_fields = len(angle_fields - group_attrs) < len(angle_fields)
        has_rotation_fields = 'orientation' in group_attrs or has_angle_fields
        if not has_rotation_fields:
            errors.append(Error(
                Error.WARNING,
                f'Group {group_name} of {group.file.filename} has no rotation fields'))
        if not has_angle_fields:
            bbp_orient_fields = {'orientation_w', 'orientation_x', 'orientation_y', 'orientation_z'}
            if 0 < len(bbp_orient_fields - group_attrs) < len(bbp_orient_fields):
                errors.append(BbpError(
                    Error.WARNING,
                    f'Group {group_name} of {group.file.filename} has no rotation fields'))

    errors = []
    group_attrs = set(group_df.columns)
    group_name = _get_group_name(group, parents=1)
    missing_fields = sorted({'morphology', 'x', 'y', 'z'} - group_attrs)
    if missing_fields:
        errors.append(fatal(f'Group {group_name} of {group.file.filename} misses '
                            f'biophysical fields: {missing_fields}'))

    errors += _check_components_dir('morphologies_dir', population)
    errors += _check_components_dir('biophysical_neuron_models_dir', population)

    morph_dirs = set()
    if 'morphologies_dir' in population:
        morph_dirs = {(population['morphologies_dir'], 'swc')}
    if 'alternate_morphologies' in population:
        for morph_type, morph_path in population['alternate_morphologies'].items():
            errors += _check_components_dir(morph_type, population['alternate_morphologies'])
            for extension, _type in EXTENSIONS_MAPPING.items():
                if _type == morph_type:
                    morph_dirs |= {(morph_path, extension)}
    if errors:
        return errors
    _check_rotations()
    for morph_path, extension in morph_dirs:
        errors += _check_files(
            f'morphology: {group_name}[{group.file.filename}]',
            (Path(morph_path, m + '.' + extension) for m in group_df['morphology']),
            Error.WARNING)
    bio_path = Path(population['biophysical_neuron_models_dir'])
    errors += _check_files(
        f'model_template: {group_name}[{group.file.filename}]',
        (bio_path / _get_model_template_file(m) for m in group_df['model_template']),
        Error.WARNING)
    return errors


def _check_nodes_group(group_df, group, config, population):
    """Validates nodes group in nodes population.

    Args:
        group_df (pd.DataFrame): nodes group in nodes .h5 file
        group (h5py.Group): nodes group in nodes .h5 file
        config (dict): resolved bluepysnap config
        population (dict): definition of current population in config

    Returns:
        list: List of errors, empty if no errors
    """
    errors = []
    if 'type' in population and population['type'] not in NODE_TYPES:
        errors.append(BbpError(Error.WARNING, f'Invalid node type: {population["type"]}'))
    if 'model_type' not in group_df.columns:
        return errors + [fatal(f'Group {_get_group_name(group, parents=1)} of '
                               f'{group.file.filename} misses "model_type" field')]
    if group_df['model_type'][0] == 'virtual':
        return errors
    if 'model_template' not in group_df.columns:
        return errors + [fatal(f'Group {_get_group_name(group, parents=1)} of '
                               f'{group.file.filename} misses "model_template" field')]
    elif group_df['model_type'][0] == 'biophysical':
        if 'components' not in config:
            return errors + [fatal('No "components" in config')]
        population = {**config['components'], **population}
        return errors + _check_bio_nodes_group(group_df, group, population)
    return errors


def _check_populations_config(populations_config, populations_h5, file_name):
    """Validates the keys (population names) in populations dictionaries.

    Args:
        populations_config (dict): edge/node populations in config nodes/edges
        populations_h5 (h5py.Group): nodes/edges group in the h5 file
        file_name (str): the name of the h5 file

    Returns:
        list: List of errors, empty if no errors
    """
    not_found = set(populations_config) - set(populations_h5)
    if not_found:
        not_found = ''.join(f'\t{p}\n' for p in not_found)
        return [fatal(f'populations not found in {file_name}:\n{not_found}')]
    return []


def _check_nodes_population(nodes_dict, config):
    """Validates nodes population.

    Args:
        nodes_dict (dict): nodes population, represented by an item of "nodes" in ``config``
        config (dict): resolved bluepysnap config

    Returns:
        list: List of errors, empty if no errors
    """
    required_datasets = ['node_type_id']
    errors = []
    nodes_file = nodes_dict.get('nodes_file')
    node_types_file = nodes_dict.get('node_types_file', None)
    with h5py.File(nodes_file, 'r') as h5f:
        nodes = _get_h5_data(h5f, 'nodes')
        if not nodes or len(nodes) == 0:
            return [fatal(f'No "nodes" in {nodes_file}.')]
        populations_config = nodes_dict.get('populations', {})
        errors += _check_populations_config(populations_config, nodes, nodes_file)
        if len(errors) > 0:
            return errors
        for population_name in nodes:
            population = nodes[population_name]
            groups = _get_population_groups(population)
            if len(groups) > 1:
                required_datasets += ['node_group_id', 'node_group_index']
            missing_datasets = sorted(set(required_datasets) - set(population))
            if missing_datasets:
                return [fatal(f'Population {population_name} of {nodes_file} misses datasets '
                              f'{missing_datasets}')]
            if len(groups) > 1:
                errors += _check_multi_groups(population['node_group_id'],
                                              population['node_group_index'], population)
                if len(errors) > 0:
                    return errors
            for group in groups:
                group_df = _nodes_group_to_dataframe(group, node_types_file, population)
                errors += _check_nodes_group(group_df, group, config,
                                             populations_config.get(population_name, {}))
    return errors


def _check_edges_group_bbp(group):
    """Validates edges group in edges population according to BBP spec.

    Not used for now. BBP only.
    Args:
        group (h5py.Group): edges group in edges .h5 file

    Returns:
        list: List of errors, empty if no errors
    """
    GROUP_NAMES = [
        'delay', 'syn_weight', 'dynamics_params',
        'afferent_section_id', 'afferent_section_pos',
        'efferent_section_id', 'efferent_section_pos',
        'afferent_center_x', 'afferent_center_y', 'afferent_center_z',
        'afferent_surface_x', 'afferent_surface_y', 'afferent_surface_z',
        'efferent_center_x', 'efferent_center_y', 'efferent_center_z',
        'efferent_surface_x', 'efferent_surface_y', 'efferent_surface_z',
    ]
    missing_fields = sorted(set(GROUP_NAMES) - set(group))
    if missing_fields:
        return [BbpError(Error.WARNING,
                         f'Group {_get_group_name(group, parents=1)} of {group.file.filename} '
                         f'misses fields: {missing_fields}')]
    return []


def _get_node_ids(node_population):
    """Gets node ids of node population.

    Args:
        node_population (h5py.Group): node population h5 instance

    Returns:
        np.ndarray: Numpy array of node ids, empty if couldn't find any
    """
    node_ids = np.empty(0)
    if 'node_id' in node_population:
        node_ids = node_population['node_id'][:]
    else:
        grp = _get_h5_data(node_population, '0')
        if grp:
            for attr in grp:
                if isinstance(grp[attr], h5py.Dataset):
                    node_ids = np.arange(len(grp[attr]))
                    break
    return node_ids


def _check_edges_node_ids(nodes_ds, nodes):
    """Checks that nodes ids in edges can be resolved to nodes ids in nodes populations.

    Args:
        nodes_ds (h5py.Dataset): nodes dataset in edges population
        nodes (list): "nodes" part of the resolved bluepysnap config

    Returns:
        list: List of errors, empty if no errors
    """
    if 'node_population' not in nodes_ds.attrs:
        return [fatal(f'Missing "node_population" attribute for "{nodes_ds.name}"')]
    node_population_name = nodes_ds.attrs['node_population']
    nodes_dict = _find_nodes_population(node_population_name, nodes)
    if not nodes_dict:
        return [fatal(f'No node population for "{nodes_ds.name}"')]
    errors = []
    with h5py.File(nodes_dict['nodes_file'], 'r') as h5f:
        node_ids = _get_node_ids(h5f['/nodes/' + node_population_name])
        if node_ids.size > 0:
            missing_ids = sorted(set(nodes_ds[:]) - set(node_ids))
            if missing_ids:
                errors.append(fatal(f'{nodes_ds.name} misses node ids in its node population: '
                                    f'{missing_ids}'))
        else:
            errors.append(fatal(f'{nodes_ds.name} does not have node ids in its node population'))
    return errors


def _check_edges_indices(population):
    """Validates edges population indices.

    Args:
        population (h5py.Group): edges population

    Returns:
        list: List of errors, empty if no errors
    """

    def _check(indices, nodes_ds):
        """The main indices check.

        It iterates over edge indices and verifies that each has its
        nodes in place in nodes populations
        """
        nodes_ranges = indices['node_id_to_ranges']
        node_to_edges_ranges = indices['range_to_edge_id']
        for node_id, nodes_range in enumerate(nodes_ranges[:]):
            if 0 <= nodes_range[0] < nodes_range[1]:
                edges_range = node_to_edges_ranges[nodes_range[0]:nodes_range[1]][0]
                edge_node_ids = list(set(nodes_ds[edges_range[0]: edges_range[1]]))
                if len(edge_node_ids) > 1 or edge_node_ids[0] != node_id:
                    errors.append(fatal(
                        f'Population {population.file.filename} edges {edge_node_ids} have '
                        f'node ids {edges_range} instead of single id {node_id}'))

    errors = []
    source_to_target = _get_h5_data(population['indices'], 'source_to_target')
    target_to_source = _get_h5_data(population['indices'], 'target_to_source')
    if not source_to_target:
        errors.append(fatal(f'No "source_to_target" in {population.file.filename}'))
    if not target_to_source:
        errors.append(fatal(f'No "target_to_source" in {population.file.filename}'))
    if target_to_source and source_to_target:
        _check(source_to_target, population['source_node_id'])
        _check(target_to_source, population['target_node_id'])
    return errors


def _check_edge_population_data(population, nodes):
    errors = []
    population_name = _get_group_name(population)
    groups = _get_population_groups(population)
    if len(groups) > 1:
        errors.append(BbpError(Error.WARNING,
                               f'Population {population_name} of {population.file.filename} have '
                               'multiple groups. Cannot be read via bluepysnap or libsonata'))

    children_object_names = set(population)
    group_datasets = ["edge_group_id", "edge_group_index"]
    required_datasets = ['edge_type_id', 'source_node_id', 'target_node_id']
    if len(groups) > 1:
        required_datasets += group_datasets

    missing_datasets = sorted(set(required_datasets) - children_object_names)
    if missing_datasets:
        return errors + [fatal(f'Population {population_name} of {population.file.filename} '
                               f'misses datasets {missing_datasets}')]
    if len(groups) == 0:
        return errors

    # pure group part
    missing_group_datasets = set(group_datasets) - set(population)
    if len(missing_group_datasets) == 1:
        return errors + [fatal(f'Population {population_name} of {population.file.filename} '
                               f'misses dataset {missing_group_datasets}')]
    if len(missing_group_datasets) == 0:
        errors += _check_multi_groups(
            population["edge_group_id"], population["edge_group_index"], population)
    if 'source_node_id' in children_object_names:
        errors += _check_edges_node_ids(population['source_node_id'], nodes)
    if 'target_node_id' in children_object_names:
        errors += _check_edges_node_ids(population['target_node_id'], nodes)
    if 'indices' in children_object_names:
        errors += _check_edges_indices(population)

    for group in groups:
        errors += _check_edges_group_bbp(group)

    return errors


def _check_edges_population(edges_dict, nodes):
    """Validates edges population.

    Args:
        edges_dict (dict): edges population, represented by an item of "edges" in ``config``
        nodes (list): "nodes" part of the resolved bluepysnap config

    Returns:
        list: List of errors, empty if no errors
    """
    errors = []
    edges_file = edges_dict.get('edges_file')
    with h5py.File(edges_file, 'r') as h5f:
        edges = _get_h5_data(h5f, 'edges')
        if not edges or len(edges) == 0:
            errors.append(fatal(f'No "edges" in {edges_file}.'))
            return errors

        populations = edges_dict.get('populations', {})
        errors += _check_populations_config(populations, edges, edges_file)
        if len(errors) > 0:
            return errors

        for populations_config in populations.values():
            if 'type' in populations_config and populations_config['type'] not in EDGE_TYPES:
                errors.append(BbpError(Error.WARNING,
                                       f'Invalid edge type: {populations_config["type"]}'))

        for population_name in edges:
            population_path = '/edges/' + population_name
            population = h5f[population_path]
            errors += _check_edge_population_data(population, nodes)

    return errors


def _check_populations(config):
    """Validates all nodes and edges populations in config.

    Args:
        config (dict): resolved bluepysnap config

    Returns:
        list: List of errors, empty if no errors
    """
    errors = []
    networks = config.get('networks')
    nodes = networks.get('nodes')
    for nodes_dict in nodes:
        errors += _check_nodes_population(nodes_dict, config)
    edges = networks.get('edges')
    for edges_dict in edges:
        errors += _check_edges_population(edges_dict, nodes)
    return errors


def validate(config_file, bbp_check=False):
    """Validates Sonata circuit.

    Args:
        config_file (str): path to Sonata circuit config file
        bbp_check (bool): whether to check BBP spec. It's additional check. It does not replace any
        official checks.

    Returns:
        list: List of errors, empty if no errors
    """
    config = Config(config_file).resolve()
    errors = _check_required_datasets(config)
    if not errors:
        errors = _check_populations(config)
    if not bbp_check:
        errors = [e for e in errors if not isinstance(e, BbpError)]
    _print_errors(errors)
    return set(errors)
