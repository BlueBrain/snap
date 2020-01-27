"""
Standalone module that validates Sonata circuit. See ``validate`` function.
"""
import itertools as it
try:
    from pathlib import Path
except:
    from pathlib2 import Path

import click
import h5py

from bluepysnap.config import Config


EXAMPLE_COUNT = 10


class ErrorContainer(object):
    class MissingDir:
        def __init__(self, name, path):
            self.name = name
            self.path = path

        def __str__(self):
            return (click.style('Invalid "{}": '.format(self.name), fg='red') +
                    str(self.path))

    class MissingFiles:
        def __init__(self, name, files):
            self.name = name
            self.files = files

        def __str__(self):
            examples = [e.name for e in it.islice(self.files, EXAMPLE_COUNT)]
            if len(self.files) > EXAMPLE_COUNT:
                examples.append('...')
            return (click.style('missing {} files in group {}:\n'.format(
                len(examples), self.name), fg='red') +
                ''.join('\t%s\n' % e for e in examples))

    def __init__(self):
        self.errors = []

    def check_dir(self, name, path):
        if not path or not Path(path).is_dir():
            self.errors.append(self.MissingDir(name, path))

    def check_files(self, name, files):
        missing = {f for f in files if not f.is_file()}
        if missing:
            self.errors.append(self.MissingFiles(name, missing))

    def fatal(self, name):
        self.errors.append(click.style(name, fg='red'))

    def print_errors(self):
        for e in self.errors:
            print(e)


def _check_components(errors, config):
    """Validates "components" part of the config.

    For now it only validates morphologies, biophysical and mechanisms dirs.
    Args:
        config (dict): resolved bluepysnap config
    """
    components = config.get('components')
    if not components:
        errors.fatal('No "components" in config')
        return

    errors.check_dir('morphologies_dir', components.get('morphologies_dir'))
    errors.check_dir('mechanisms_dir', components.get('mechanisms_dir'))
    errors.check_dir('biophysical_neuron_models_dir',
                     components.get('biophysical_neuron_models_dir'))


def _check_required_datasets(errors, config):
    """Validates required datasets of "nodes" and "edges" in config

    Args:
        config (dict): resolved bluepysnap config

    Returns:
        True if everything is fine, otherwise False.
    """
    networks = config.get('networks')
    if not networks:
        errors.fatal('No "networks" in config')
        return False
    nodes = networks.get('nodes')
    if not nodes:
        errors.fatal('No "nodes" in config "networks"')
    edges = networks.get('edges')
    if not edges:
        errors.fatal.warn('No "edges" in config "networks"')
    if not nodes or not edges:
        return False

    for nodes_dict in nodes:
        nodes_file = nodes_dict.get('nodes_file')
        if nodes_file is None or not Path(nodes_file).is_file():
            errors.fatal('Invalid "nodes_file": {}'.format(nodes_file))
        types_file = nodes_dict.get('node_types_file')
        if types_file is not None and not Path(types_file).is_file():
            errors.fatal('Invalid "node_types_file": {}'.format(types_file))

    for edges_dict in edges:
        edges_file = edges_dict.get('edges_file')
        if edges_file is None or not Path(edges_file).is_file():
            errors.fatal('Invalid "edges_file": {}'.format(edges_file))
        types_file = edges_dict.get('edge_types_file')
        if types_file is not None and not Path(types_file).is_file():
            errors.fatal('Invalid "edge_types_file": {}'.format(types_file))

    return True


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


def _check_nodes_group(errors, group, config):
    """Validates nodes group in nodes population

    Args:
        group (h5py.Group): nodes group in nodes .h5 file
        config (dict): resolved bluepysnap config
    """
    # TODO: decide what are optional/non-optional

    GROUP_NAMES = ['model_type',  # Has 4 valid values: biophysical, virtual, single_compartment, and point_neuron
                   'model_template',
                   'dynamics_params',
                   ]

    BIO_GROUP_NAMES = ['morphology', 'orientation',
                       'rotation_angle_xaxis',
                       'rotation_angle_yaxis',
                       'rotation_angle_zaxis',
                       'x', 'y', 'z',
                       'dynamics_params']

    OPTIONAL_GROUP_NAMES = ['recenter',
                            ]

    group_name = _get_group_name(group)

    missing_fields = set(GROUP_NAMES + BIO_GROUP_NAMES + OPTIONAL_GROUP_NAMES) - set(group)
    if missing_fields:
        errors.fatal('Group {} of {} misses fields: {}'.format(
            group_name, group.file.filename, missing_fields))

    errors.check_files('morphology: {}[{}]'.format(group_name, group.file.filename),
                       (Path(config['components']['morphologies_dir'], m)
                        for m in group.get('morphology')))

    errors.check_files('model_template: {}[{}]'.format(group_name, group.file.filename),
                       {Path(config['components']['biophysical_neuron_models_dir'],
                             m.split('hoc:')[1])
                        for m in group.get('model_template')})


def _check_nodes_population(errors, nodes_dict, config):
    """Validates nodes population

    Args:
        nodes_dict (dict): nodes population, represented by an item of "nodes" in ``config``
        config (dict): resolved bluepysnap config
    """
    POPULATION_DATASET_NAMES = ['node_type_id', 'node_id', 'node_group_id', 'node_group_index']
    nodes_file = nodes_dict.get('nodes_file')
    with h5py.File(nodes_file, 'r') as f:
        nodes = f.get('nodes')
        if not nodes or len(nodes) == 0:
            errors.fatal('No "nodes" in {}.'.format(nodes_file))
        for population_name in nodes:
            population = nodes[population_name]
            children_names = population
            missing_datasets = set(POPULATION_DATASET_NAMES) - set(children_names)
            if missing_datasets:
                errors.fatal('Population {} of {} misses datasets {}'.format(
                    population_name, nodes_file, missing_datasets))
            for name in children_names:
                if isinstance(population[name], h5py.Group):
                    _check_nodes_group(errors, population[name], config)


def _check_edges_group(errors, group):
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

    missing_fields = set(GROUP_NAMES) - set(group)
    if len(missing_fields) > 0:
        errors.fatal('Group {} of {} misses fields: {}'.format(
            _get_group_name(group), group.file.filename, missing_fields))


def _check_edges_node_ids(errors, nodes_ds, nodes):
    """Validates that nodes ids in edges can be resolved to nodes ids in nodes populations

    Args:
        nodes_ds: nodes dataset in edges population
        nodes (list): "nodes" part of the resolved bluepysnap config
    """
    node_population_name = nodes_ds.attrs['node_population']
    nodes_dict = _find_nodes_population(node_population_name, nodes)
    if not nodes_dict:
        errors.fatal('No node population for "{}"'.format(nodes_ds.name))
        return
    with h5py.File(nodes_dict['nodes_file'], 'r') as f:
        node_population = f['/nodes/' + node_population_name]
        if 'node_id' in node_population:
            node_ids = node_population['node_id'][:]
        elif 'node_type_id' in node_population:
            node_ids = range(len(node_population['node_type_id']))
        else:
            errors.fatal('{} does not have node ids in its node population'.format(nodes_ds.name))
            return
    missing_ids = set(nodes_ds[:]) - set(node_ids)
    if missing_ids:
        errors.fatal('{} misses node ids in its node population: {}'.format(
            nodes_ds.name, missing_ids))


def _check_edges_indices(errors, population):
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
                    errors.fatal(
                        'Population {} edges {} have node ids {} instead of single id {}'.format(
                            population.file.filename, edge_node_ids, edges_range, node_id))

    source_to_target = population['indices'].get('source_to_target')
    target_to_source = population['indices'].get('target_to_source')
    if not source_to_target:
        errors.fatal('No "source_to_target" in {}'.format(population.file.filename))
    if not target_to_source:
        errors.fatal('No "target_to_source" in {}'.format(population.file.filename))
    if target_to_source and source_to_target:
        _check(source_to_target, population['source_node_id'])
        _check(target_to_source, population['target_node_id'])


def _check_edges_population(errors, edges_dict, nodes):
    """Validates edges population

    Args:
        edges_dict (dict): edges population, represented by an item of "edges" in ``config``
        nodes (list): "nodes" part of the resolved bluepysnap config
    """
    POPULATION_DATASET_NAMES = [
        'edge_type_id', 'source_node_id', 'target_node_id', 'edge_group_id', 'edge_group_index']
    edges_file = edges_dict.get('edges_file')
    with h5py.File(edges_file, 'r') as f:
        edges = f.get('edges')
        if not edges or len(edges) == 0:
            errors.fatal('No "edges" in {}.'.format(edges_file))
        for population_name in edges:
            population_path = '/edges/' + population_name
            population = f[population_path]
            children_names = population.keys()
            missing_datasets = set(POPULATION_DATASET_NAMES) - set(children_names)
            if len(missing_datasets) > 0:
                errors.fatal('Population {} of {} misses datasets {}'.format(
                    population_name, edges_file, missing_datasets))
            for name in children_names - {'indices'}:
                if isinstance(population[name], h5py.Group):
                    _check_edges_group(errors, population[name])
            if 'source_node_id' in children_names:
                _check_edges_node_ids(errors, population['source_node_id'], nodes)
            if 'target_node_id' in children_names:
                _check_edges_node_ids(errors, population['target_node_id'], nodes)
            if 'indices' in children_names:
                _check_edges_indices(errors, population)


def _check_populations(errors, config):
    """Validates all nodes and edges populations in config

    Args:
        config (dict): resolved bluepysnap config
    """
    networks = config.get('networks')
    nodes = networks.get('nodes')
    for nodes_dict in nodes:
        _check_nodes_population(errors, nodes_dict, config)
    edges = networks.get('edges')
    for edges_dict in edges:
        _check_edges_population(errors, edges_dict, nodes)


def validate(config_file):
    """Validates Sonata circuit

    Args:
        config_file (str): path to Sonata circuit config file
    """
    errors = ErrorContainer()
    config = Config(config_file).resolve()
    _check_components(errors, config)
    if _check_required_datasets(errors, config):
        _check_populations(errors, config)
    errors.print_errors()
