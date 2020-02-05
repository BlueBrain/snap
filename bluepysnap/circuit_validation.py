"""Standalone module that validates Sonata circuit. See ``validate`` function."""
import itertools as it
import numpy as np
try:
    from pathlib import Path
except ImportError:  # pragma: nocover
    from pathlib2 import Path
import click
import h5py

from bluepysnap.config import Config

MAX_MISSING_FILES_DISPLAY = 10


class Error(object):
    """Error used for reporting of validation errors."""
    FATAL = 'FATAL'
    WARNING = 'WARNING'

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
        return [fatal('Invalid components "{}": {}'.format(name, dirpath))]
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
        filenames = ''.join('\t%s\n' % e for e in examples)
        return [Error(level,
                      'missing {} files in group {}:\n{}'.format(len(missing), name, filenames))]
    return []


def _print_errors(errors):
    """Some fancy errors printing."""
    colors = {
        Error.WARNING: 'yellow',
        Error.FATAL: 'red',
    }
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
            errors.append(fatal('Invalid "nodes_file": {}'.format(nodes_file)))
        types_file = nodes_dict.get('node_types_file')
        if types_file is not None and not Path(types_file).is_file():
            errors.append(fatal('Invalid "node_types_file": {}'.format(types_file)))

    for edges_dict in edges:
        edges_file = edges_dict.get('edges_file')
        if edges_file is None or not Path(edges_file).is_file():
            errors.append(fatal('Invalid "edges_file": {}'.format(edges_file)))
        types_file = edges_dict.get('edge_types_file')
        if types_file is not None and not Path(types_file).is_file():
            errors.append(fatal('Invalid "edge_types_file": {}'.format(types_file)))

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


def _get_group_name(group):
    """Gets group name of h5 group."""
    return Path(group.parent.name).name


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


def _check_bio_nodes_group(group, config):
    """Checks biophysical nodes group for errors.

    Args:
        group (h5py.Group): nodes group in nodes .h5 file
        config (dict): resolved bluepysnap config

    Returns:
        list: List of errors, empty if no errors
    """

    def _check_rotations():
        """Checks for proper rotation fields."""
        angle_fields = {'rotation_angle_xaxis', 'rotation_angle_yaxis', 'rotation_angle_zaxis'}
        has_angle_fields = len(angle_fields - set(group)) < len(angle_fields)
        has_rotation_fields = 'orientation' in group or has_angle_fields
        if not has_rotation_fields:
            errors.append(Error(Error.WARNING, 'Group {} of {} has no rotation fields'.
                                format(group_name, group.file.filename)))
        if not has_angle_fields:
            bbp_orient_fields = {'orientation_w', 'orientation_x', 'orientation_y', 'orientation_z'}
            if 0 < len(bbp_orient_fields - set(group)) < len(bbp_orient_fields):
                errors.append(BbpError(Error.WARNING, 'Group {} of {} has no rotation fields'.
                                       format(group_name, group.file.filename)))

    errors = []
    group_name = _get_group_name(group)
    missing_fields = sorted({'morphology', 'x', 'y', 'z'} - set(group))
    if missing_fields:
        errors.append(fatal('Group {} of {} misses biophysical fields: {}'.
                            format(group_name, group.file.filename, missing_fields)))
    _check_rotations()
    components = config['components']
    errors += _check_components_dir('morphologies_dir', components)
    errors += _check_components_dir('mechanisms_dir', components)
    errors += _check_components_dir('biophysical_neuron_models_dir', components)
    if errors:
        return errors
    errors += _check_files(
        'morphology: {}[{}]'.format(group_name, group.file.filename),
        (Path(components['morphologies_dir'], m + '.swc') for m in group['morphology']),
        Error.WARNING)
    bio_path = Path(components['biophysical_neuron_models_dir'])
    errors += _check_files(
        'model_template: {}[{}]'.format(group_name, group.file.filename),
        (bio_path / _get_model_template_file(m) for m in group['model_template']),
        Error.WARNING)
    return errors


def _check_nodes_group(group, config):
    """Validates nodes group in nodes population.

    Args:
        group (h5py.Group): nodes group in nodes .h5 file
        config (dict): resolved bluepysnap config

    Returns:
        list: List of errors, empty if no errors
    """
    REQUIRED_GROUP_NAMES = ['model_type', 'model_template']
    missing_fields = sorted(set(REQUIRED_GROUP_NAMES) - set(group))
    if missing_fields:
        return [fatal('Group {} of {} misses required fields: {}'
                      .format(_get_group_name(group), group.file.filename, missing_fields))]
    elif 'biophysical' in group['model_type'][:]:
        return _check_bio_nodes_group(group, config)
    return []


def _check_nodes_population(nodes_dict, config):
    """Validates nodes population.

    Args:
        nodes_dict (dict): nodes population, represented by an item of "nodes" in ``config``
        config (dict): resolved bluepysnap config

    Returns:
        list: List of errors, empty if no errors
    """
    POPULATION_DATASET_NAMES = ['node_type_id', 'node_group_id', 'node_group_index']
    errors = []
    nodes_file = nodes_dict.get('nodes_file')
    with h5py.File(nodes_file, 'r') as h5f:
        nodes = _get_h5_data(h5f, 'nodes')
        if not nodes or len(nodes) == 0:
            errors.append(fatal('No "nodes" in {}.'.format(nodes_file)))
            return errors
        for population_name in nodes:
            population = nodes[population_name]
            missing_datasets = sorted(set(POPULATION_DATASET_NAMES) - set(population))
            if missing_datasets:
                errors.append(fatal('Population {} of {} misses datasets {}'.
                                    format(population_name, nodes_file, missing_datasets)))
            for name in population:
                if isinstance(population[name], h5py.Group):
                    errors += _check_nodes_group(population[name], config)
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
        return [BbpError(Error.WARNING, 'Group {} of {} misses fields: {}'.
                         format(_get_group_name(group), group.file.filename, missing_fields))]
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
        node_size_ds = _get_h5_data(node_population, 'node_type_id') \
            or _get_h5_data(node_population, 'node_group_id')
        if node_size_ds:
            node_ids = np.arange(len(node_size_ds))
    return node_ids


def _check_edges_node_ids(nodes_ds, nodes):
    """Checks that nodes ids in edges can be resolved to nodes ids in nodes populations.

    Args:
        nodes_ds (h5py.Dataset): nodes dataset in edges population
        nodes (list): "nodes" part of the resolved bluepysnap config

    Returns:
        list: List of errors, empty if no errors
    """
    errors = []
    node_population_name = nodes_ds.attrs['node_population']
    nodes_dict = _find_nodes_population(node_population_name, nodes)
    if not nodes_dict:
        errors.append(fatal('No node population for "{}"'.format(nodes_ds.name)))
        return errors
    with h5py.File(nodes_dict['nodes_file'], 'r') as h5f:
        node_ids = _get_node_ids(h5f['/nodes/' + node_population_name])
        if node_ids.size > 0:
            missing_ids = sorted(set(nodes_ds[:]) - set(node_ids))
            if missing_ids:
                errors.append(fatal('{} misses node ids in its node population: {}'.
                                    format(nodes_ds.name, missing_ids)))
        else:
            errors.append(fatal('{} does not have node ids in its node population'.
                                format(nodes_ds.name)))
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
                        'Population {} edges {} have node ids {} instead of single id {}'.format(
                            population.file.filename, edge_node_ids, edges_range, node_id)))

    errors = []
    source_to_target = _get_h5_data(population['indices'], 'source_to_target')
    target_to_source = _get_h5_data(population['indices'], 'target_to_source')
    if not source_to_target:
        errors.append(fatal('No "source_to_target" in {}'.format(population.file.filename)))
    if not target_to_source:
        errors.append(fatal('No "target_to_source" in {}'.format(population.file.filename)))
    if target_to_source and source_to_target:
        _check(source_to_target, population['source_node_id'])
        _check(target_to_source, population['target_node_id'])
    return errors


def _check_edges_population(edges_dict, nodes):
    """Validates edges population.

    Args:
        edges_dict (dict): edges population, represented by an item of "edges" in ``config``
        nodes (list): "nodes" part of the resolved bluepysnap config

    Returns:
        list: List of errors, empty if no errors
    """
    POPULATION_DATASET_NAMES = [
        'edge_type_id', 'source_node_id', 'target_node_id', 'edge_group_id', 'edge_group_index']
    errors = []
    edges_file = edges_dict.get('edges_file')
    with h5py.File(edges_file, 'r') as h5f:
        edges = _get_h5_data(h5f, 'edges')
        if not edges or len(edges) == 0:
            errors.append(fatal('No "edges" in {}.'.format(edges_file)))
            return errors
        for population_name in edges:
            population_path = '/edges/' + population_name
            population = h5f[population_path]
            children_names = set(population.keys())
            missing_datasets = sorted(set(POPULATION_DATASET_NAMES) - children_names)
            if missing_datasets:
                errors.append(fatal('Population {} of {} misses datasets {}'.
                                    format(population_name, edges_file, missing_datasets)))
                return errors
            for name in children_names - {'indices'}:
                if isinstance(population[name], h5py.Group):
                    errors += _check_edges_group_bbp(population[name])
            if 'source_node_id' in children_names:
                errors += _check_edges_node_ids(population['source_node_id'], nodes)
            if 'target_node_id' in children_names:
                errors += _check_edges_node_ids(population['target_node_id'], nodes)
            if 'indices' in children_names:
                errors += _check_edges_indices(population)
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
    if 'components' not in config:
        errors.append(fatal('No "components" in config'))
    if not errors:
        errors = _check_populations(config)
    if not bbp_check:
        errors = [e for e in errors if not isinstance(e, BbpError)]
    _print_errors(errors)
    return errors
