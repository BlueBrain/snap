"""
Standalone module that validates Sonata circuit. See ``validate`` function.
"""
import itertools as it
try:
    from pathlib import Path
except:
    from pathlib2 import Path
from enum import Enum
import click
import h5py

from bluepysnap.config import Config


class ErrorLevel(Enum):
    FATAL = 0
    WARNING = 1


class Error(object):
    def __init__(self, level, message=None):
        self.level = level
        self.message = message

    def __str__(self):
        return str(self.message)


class MissingDirError(Error):
    def __init__(self, name, path, level, message=None):
        super().__init__(level, message)
        self.name = name
        self.path = path

    def __str__(self):
        return click.style('Invalid "{}": '.format(self.name), fg='red') + str(self.path)


class MissingFiles(Error):
    MAX_COUNT = 10

    def __init__(self, name, files, level, message=None):
        super().__init__(level, message)
        self.name = name
        self.files = files

    def __str__(self):
        examples = [e.name for e in it.islice(self.files, self.MAX_COUNT)]
        if len(self.files) > self.MAX_COUNT:
            examples.append('...')
        return click.style('missing {} files in group {}:\n'.format(
            len(examples), self.name), fg='red') + ''.join('\t%s\n' % e for e in examples)


def check_dir(name, path, level):
    if not path or not Path(path).is_dir():
        return [MissingDirError(name, path, level)]
    return []


def check_files(name, files, level):
    missing = {f for f in files if not f.is_file()}
    if missing:
        return [MissingFiles(name, missing, level)]
    return []


def fatal(message):
    return Error(ErrorLevel.FATAL, message)


def warn(message):
    return Error(ErrorLevel.WARNING, message)


def print_errors(errors):
    colors = {
        ErrorLevel.WARNING: 'yellow',
        ErrorLevel.FATAL: 'red',
    }
    for error in errors:
        print(click.style(error.level.name + ': ', fg=colors[error.level]) + str(error))


def _check_components(config):
    """Validates "components" part of the config.

    For now it only validates morphologies, biophysical and mechanisms dirs.
    Args:
        config (dict): resolved bluepysnap config
    """
    errors = []
    components = config.get('components')
    if not components:
        errors.append(Error(ErrorLevel.FATAL, 'No "components" in config'))
        return errors

    errors += check_dir('morphologies_dir', components.get('morphologies_dir'), ErrorLevel.WARNING)
    errors += check_dir('mechanisms_dir', components.get('mechanisms_dir'), ErrorLevel.WARNING)
    errors += check_dir('biophysical_neuron_models_dir',
                        components.get('biophysical_neuron_models_dir'), ErrorLevel.WARNING)
    return errors


def _check_required_datasets(config):
    """Validates required datasets of "nodes" and "edges" in config

    Args:
        config (dict): resolved bluepysnap config

    Returns:
        True if everything is fine, otherwise False.
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
        A dict item of "nodes" of the resolved bluepysnap config. None if didn't find anything.
    """
    for nodes_dict in nodes:
        nodes_file = nodes_dict.get('nodes_file')
        if nodes_file:
            with h5py.File(nodes_file, 'r') as f:
                if '/nodes/' + node_population_name in f:
                    return nodes_dict
    return None


def _get_group_name(group):
    return Path(group.parent.name).name


def _get_model_template_file(model_template):
    parts = model_template.split(':', 1)
    return parts[1] + '.' + parts[0]


def _check_nodes_group(group, config):
    """Validates nodes group in nodes population

    Args:
        group (h5py.Group): nodes group in nodes .h5 file
        config (dict): resolved bluepysnap config
    """
    REQUIRED_GROUP_NAMES = ['model_type', 'model_template']
    errors = []
    group_name = _get_group_name(group)

    missing_fields = set(REQUIRED_GROUP_NAMES) - set(group)
    if missing_fields:
        errors.append(
            fatal('Group {} of {} misses required fields: {}'
                  .format(group_name, group.file.filename, missing_fields)))
        return errors

    if 'biophysical' in group['model_type'][:]:
        missing_fields = {'morphology', 'x', 'y', 'z'} - set(group)
        if missing_fields:
            errors.append(fatal('Group {} of {} misses biophysical fields: {}'.
                                format(group_name, group.file.filename, missing_fields)))
            return errors

        has_rotation_fields = 'orientation' in group \
                              or {'rotation_angle_xaxis', 'rotation_angle_yaxis',
                                  'rotation_angle_zaxis'} - set(group) is not None
        if not has_rotation_fields:
            errors.append(fatal('Group {} of {} has no rotation fields'.
                                format(group_name, group.file.filename)))

        morph_path = Path(config['components']['morphologies_dir'])
        errors += check_files(
            'morphology: {}[{}]'.format(group_name, group.file.filename),
            (morph_path / (m + '.swc') for m in group['morphology']),
            ErrorLevel.WARNING)
        bio_path = Path(config['components']['biophysical_neuron_models_dir'])
        errors += check_files(
            'model_template: {}[{}]'.format(group_name, group.file.filename),
            (bio_path / _get_model_template_file(m) for m in group['model_template']),
            ErrorLevel.WARNING)
    return errors


def _check_nodes_population(nodes_dict, config):
    """Validates nodes population

    Args:
        nodes_dict (dict): nodes population, represented by an item of "nodes" in ``config``
        config (dict): resolved bluepysnap config
    """
    POPULATION_DATASET_NAMES = ['node_type_id', 'node_group_id', 'node_group_index']
    errors = []
    nodes_file = nodes_dict.get('nodes_file')
    with h5py.File(nodes_file, 'r') as f:
        nodes = f.get('nodes')
        if not nodes or len(nodes) == 0:
            errors.append(fatal('No "nodes" in {}.'.format(nodes_file)))
        for population_name in nodes:
            population = nodes[population_name]
            children_names = population
            missing_datasets = set(POPULATION_DATASET_NAMES) - set(children_names)
            if missing_datasets:
                errors.append(fatal('Population {} of {} misses datasets {}'.
                                    format(population_name, nodes_file, missing_datasets)))
            if 'node_id' not in population:
                errors.append(warn('Population {} of {} misses "node_id" dataset'.
                                   format(population_name, nodes_file)))
            for name in children_names:
                if isinstance(population[name], h5py.Group):
                    errors += _check_nodes_group(population[name], config)
    return errors


def _check_edges_group_bbp(group):
    """Validates edges group in edges population

    Args:
        group (h5py.Group): edges group in edges .h5 file
    """
    GROUP_NAMES = [
        'delay', 'syn_weight', 'model_template', 'dynamics_params',
        'afferent_section_id', 'afferent_section_pos',
        'efferent_section_id', 'efferent_section_pos',
        'afferent_center_x', 'afferent_center_y', 'afferent_center_z',
        'afferent_surface_x', 'afferent_surface_y', 'afferent_surface_z',
        'efferent_center_x', 'efferent_center_y', 'efferent_center_z',
        'efferent_surface_x', 'efferent_surface_y', 'efferent_surface_z',
    ]
    errors = []
    missing_fields = set(GROUP_NAMES) - set(group)
    if len(missing_fields) > 0:
        errors.append(fatal('Group {} of {} misses fields: {}'.
                            format(_get_group_name(group), group.file.filename, missing_fields)))
    return errors


def _check_edges_node_ids(nodes_ds, nodes):
    """Validates that nodes ids in edges can be resolved to nodes ids in nodes populations

    Not used for now. BBP only.
    Args:
        nodes_ds: nodes dataset in edges population
        nodes (list): "nodes" part of the resolved bluepysnap config
    """
    errors = []
    node_population_name = nodes_ds.attrs['node_population']
    nodes_dict = _find_nodes_population(node_population_name, nodes)
    if not nodes_dict:
        errors.append(fatal('No node population for "{}"'.format(nodes_ds.name)))
        return errors
    with h5py.File(nodes_dict['nodes_file'], 'r') as f:
        node_population = f['/nodes/' + node_population_name]
        if 'node_id' in node_population:
            node_ids = node_population['node_id'][:]
        elif 'node_type_id' in node_population:
            node_ids = range(len(node_population['node_type_id']))
        else:
            errors.append(fatal('{} does not have node ids in its node population'.
                                format(nodes_ds.name)))
            return errors
    missing_ids = set(nodes_ds[:]) - set(node_ids)
    if missing_ids:
        errors.append(fatal('{} misses node ids in its node population: {}'.
                            format(nodes_ds.name, missing_ids)))
    return errors


def _check_edges_indices(population):
    """Validates edges population indices

    Args:
        population (h5py.Group): edges population
    """

    def _check(indices, nodes_ds):
        nodes_ranges = indices['node_id_to_ranges']
        node_to_edges_ranges = indices['range_to_edge_id']
        for node_id, nodes_range in enumerate(nodes_ranges[:]):
            if 0 <= nodes_range[0] < nodes_range[1]:
                edges_range = node_to_edges_ranges[nodes_range[0]:nodes_range[1]][0]
                edge_node_ids = set(nodes_ds[edges_range[0]: edges_range[1]])
                if len(edge_node_ids) > 1 or edge_node_ids.pop() != node_id:
                    errors.append(fatal(
                        'Population {} edges {} have node ids {} instead of single id {}'.format(
                            population.file.filename, edge_node_ids, edges_range, node_id)))

    errors = []
    source_to_target = population['indices'].get('source_to_target')
    target_to_source = population['indices'].get('target_to_source')
    if not source_to_target:
        errors.append(fatal('No "source_to_target" in {}'.format(population.file.filename)))
    if not target_to_source:
        errors.append(fatal('No "target_to_source" in {}'.format(population.file.filename)))
    if target_to_source and source_to_target:
        _check(source_to_target, population['source_node_id'])
        _check(target_to_source, population['target_node_id'])
    return errors


def _check_edges_population(edges_dict, nodes):
    """Validates edges population

    Args:
        edges_dict (dict): edges population, represented by an item of "edges" in ``config``
        nodes (list): "nodes" part of the resolved bluepysnap config
    """
    POPULATION_DATASET_NAMES = [
        'edge_type_id', 'source_node_id', 'target_node_id', 'edge_group_id', 'edge_group_index']
    errors = []
    edges_file = edges_dict.get('edges_file')
    with h5py.File(edges_file, 'r') as f:
        edges = f.get('edges')
        if not edges or len(edges) == 0:
            errors.append(fatal('No "edges" in {}.'.format(edges_file)))
        for population_name in edges:
            population_path = '/edges/' + population_name
            population = f[population_path]
            children_names = population.keys()
            missing_datasets = set(POPULATION_DATASET_NAMES) - set(children_names)
            if len(missing_datasets) > 0:
                errors.append(fatal('Population {} of {} misses datasets {}'.
                                    format(population_name, edges_file, missing_datasets)))
            if 'edge_id' not in population:
                errors.append(warn('Population {} of {} misses "edge_id" dataset'.
                                   format(population_name, edges_file)))
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
    """Validates all nodes and edges populations in config

    Args:
        config (dict): resolved bluepysnap config
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


def validate(config_file):
    """Validates Sonata circuit

    Args:
        config_file (str): path to Sonata circuit config file
    """
    config = Config(config_file).resolve()
    errors = _check_required_datasets(config)
    if not errors:
        errors = _check_populations(config)
    errors += _check_components(config)
    print_errors(errors)
    return errors
