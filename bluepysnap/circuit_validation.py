"""
Standalone module that validates Sonata circuit. See ``validate`` function.
"""
import warnings
from pathlib2 import Path

import h5py

from bluepysnap.config import Config


def _check_components(config):
    """Validates "components" part of the config.

    For now it only validates morphologies, biophysical and mechanisms dirs.
    Args:
        config (dict): resolved bluepysnap config
    """
    components = config.get('components')
    if not components:
        warnings.warn('No "components" in config')
        return
    morphologies = components.get('morphologies_dir')
    if not morphologies or not Path(morphologies).is_dir():
        warnings.warn('Invalid "morphologies_dir": {}'.format(morphologies))
    mechanisms = components.get('mechanisms_dir')
    if not mechanisms or not Path(mechanisms).is_dir():
        warnings.warn('Invalid "mechanisms_dir": {}'.format(mechanisms))
    biophysics = components.get('biophysical_neuron_models_dir')
    if not biophysics or not Path(biophysics).is_dir():
        warnings.warn('Invalid "biophysical_neuron_models_dir": {}'.format(biophysics))


def _check_required_datasets(config):
    """Validates required datasets of "nodes" and "edges" in config

    Args:
        config (dict): resolved bluepysnap config

    Returns:
        True if everything is fine, otherwise False. Errors are printed as warnings.
    """
    networks = config.get('networks')
    if not networks:
        warnings.warn('No "networks" in config')
        return False
    nodes = networks.get('nodes')
    if not nodes:
        warnings.warn('No "nodes" in config "networks"')
    edges = networks.get('edges')
    if not edges:
        warnings.warn('No "edges" in config "networks"')
    if not nodes or not edges:
        return False

    for nodes_dict in nodes:
        nodes_file = nodes_dict.get('nodes_file')
        if nodes_file is None or not Path(nodes_file).is_file():
            warnings.warn('Invalid "nodes_file": {}'.format(nodes_file))
        types_file = nodes_dict.get('node_types_file')
        if types_file is not None and not Path(types_file).is_file():
            warnings.warn('Invalid "node_types_file": {}'.format(types_file))
    for edges_dict in edges:
        edges_file = edges_dict.get('edges_file')
        if edges_file is None or not Path(edges_file).is_file():
            warnings.warn('Invalid "edges_file": {}'.format(edges_file))
        types_file = edges_dict.get('edge_types_file')
        if types_file is not None and not Path(types_file).is_file():
            warnings.warn('Invalid "edge_types_file": {}'.format(types_file))
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
            with h5py.File(nodes_file) as f:
                if '/nodes/' + node_population_name in f:
                    return nodes_dict
    return None


def _check_nodes_group(group, config):
    """Validates nodes group in nodes population

    Any errors are printed as warnings

    Args:
        group (h5py.Group): nodes group in nodes .h5 file
        config (dict): resolved bluepysnap config
    """
    GROUP_NAMES = [
        'model_type', 'model_template', 'morphology', 'orientation',
        'rotation_angle_xaxis', 'rotation_angle_yaxis', 'rotation_angle_zaxis',
        'x', 'y', 'z', 'recenter', 'dynamics_params']

    missing_fields = set(GROUP_NAMES) - set(group.keys())
    if len(missing_fields) > 0:
        warnings.warn('Group {} of {} misses fields: {}'.format(
            group, group.file.filename, missing_fields))
    morphology_ds = group.get('morphology')
    if morphology_ds:
        for morphology in morphology_ds[:]:
            if not Path(config['components']['morphologies_dir'], morphology).is_file():
                warnings.warn('non-existent morphology file {} in group {} of {}'.format(
                    morphology, group, group.file.filename
                ))
    model_template_ds = group.get('model_template')
    if model_template_ds:
        for model_template in model_template_ds[:]:
            filename = model_template.split('hoc:')[1]
            if not Path(config['components']['biophysical_neuron_models_dir'], filename).is_file():
                warnings.warn('non-existent model_template file {} in group {} of {}'.format(
                    filename, group, group.file.filename
                ))


def _check_nodes_population(nodes_dict, config):
    """Validates nodes population

    Any errors are printed as warnings

    Args:
        nodes_dict (dict): nodes population, represented by an item of "nodes" in ``config``
        config (dict): resolved bluepysnap config
    """
    POPULATION_DATASET_NAMES = ['node_type_id', 'node_id', 'node_group_id', 'node_group_index']
    nodes_file = nodes_dict.get('nodes_file')
    with h5py.File(nodes_file) as f:
        nodes = f.get('nodes')
        if not nodes or len(nodes.keys()) == 0:
            warnings.warn('No "nodes" in {}.'.format(nodes_file))
        for population_name in nodes.keys():
            population = nodes[population_name]
            children_names = population.keys()
            missing_datasets = set(POPULATION_DATASET_NAMES) - set(children_names)
            if len(missing_datasets) > 0:
                warnings.warn('Population {} of {} misses datasets {}'.format(
                    population, nodes_file, missing_datasets))
            for name in children_names:
                if isinstance(population[name], h5py.Group):
                    _check_nodes_group(population[name], config)


def _check_edges_group(group):
    """Validates edges group in edges population

    Any errors are printed as warnings

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

    missing_fields = set(GROUP_NAMES) - set(group.keys())
    if len(missing_fields) > 0:
        warnings.warn('Group {} of {} misses fields: {}'.format(
            group, group.file.filename, missing_fields))


def _check_edges_node_ids(nodes_ds, nodes):
    """Validates that nodes ids in edges can be resolved to nodes ids in nodes populations

    Any errors are printed as warnings

    Args:
        nodes_ds: nodes dataset in edges population
        nodes (list): "nodes" part of the resolved bluepysnap config
    """
    node_population_name = nodes_ds.attrs['node_population']
    nodes_dict = _find_nodes_population(node_population_name, nodes)
    if not nodes_dict:
        warnings.warn('No node population for "{}"'.format(nodes_ds.name))
        return
    with h5py.File(nodes_dict['nodes_file']) as f:
        node_population = f['/nodes/' + node_population_name]
        if 'node_id' in node_population:
            node_ids = node_population['node_id'][:]
        elif 'node_type_id' in node_population:
            node_ids = range(len(node_population['node_type_id']))
        else:
            warnings.warn('{} does not have node ids in its node population'.format(nodes_ds.name))
            return
    missing_ids = set(nodes_ds[:]) - set(node_ids)
    if missing_ids:
        warnings.warn('{} misses node ids in its node population: {}'.format(
            nodes_ds.name, missing_ids))


def _check_edges_indices(population):
    """Validates edges population indices

    Any errors are printed as warnings

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
                    warnings.warn(
                        'Population {} edges {} have node ids {} instead of single id {}'.format(
                            population.file.filename, edge_node_ids, edges_range, node_id))

    source_to_target = population['indices'].get('source_to_target')
    target_to_source = population['indices'].get('target_to_source')
    if not source_to_target:
        warnings.warn('No "source_to_target" in {}'.format(population.file.filename))
    if not target_to_source:
        warnings.warn('No "target_to_source" in {}'.format(population.file.filename))
    if target_to_source and source_to_target:
        _check(source_to_target, population['source_node_id'])
        _check(target_to_source, population['target_node_id'])


def _check_edges_population(edges_dict, nodes):
    """Validates edges population

    Any errors are printed as warnings

    Args:
        edges_dict (dict): edges population, represented by an item of "edges" in ``config``
        nodes (list): "nodes" part of the resolved bluepysnap config
    """
    POPULATION_DATASET_NAMES = [
        'edge_type_id', 'source_node_id', 'target_node_id', 'edge_group_id', 'edge_group_index']
    edges_file = edges_dict.get('edges_file')
    with h5py.File(edges_file) as f:
        edges = f.get('edges')
        if not edges or len(edges.keys()) == 0:
            warnings.warn('No "edges" in {}.'.format(edges_file))
        for population_name in edges.keys():
            population_path = '/edges/' + population_name
            population = f[population_path]
            children_names = population.keys()
            missing_datasets = set(POPULATION_DATASET_NAMES) - set(children_names)
            if len(missing_datasets) > 0:
                warnings.warn('Population {} of {} misses datasets {}'.format(
                    population, edges_file, missing_datasets))
            for name in children_names - {'indices'}:
                if isinstance(population[name], h5py.Group):
                    _check_edges_group(population[name])
            if 'source_node_id' in children_names:
                _check_edges_node_ids(population['source_node_id'], nodes)
            if 'target_node_id' in children_names:
                _check_edges_node_ids(population['target_node_id'], nodes)
            if 'indices' in children_names:
                _check_edges_indices(population)


def _check_populations(config):
    """Validates all nodes and edges populations in config

    Any errors are printed as warnings

    Args:
        config (dict): resolved bluepysnap config
    """
    networks = config.get('networks')
    nodes = networks.get('nodes')
    for nodes_dict in nodes:
        _check_nodes_population(nodes_dict, config)
    edges = networks.get('edges')
    for edges_dict in edges:
        _check_edges_population(edges_dict, nodes)


def validate(config_file):
    """Validates Sonata circuit

    In case any problems warnings are printed

    Args:
        config_file (str): path to Sonata circuit config file
    """
    config = Config(config_file).resolve()
    _check_components(config)
    if _check_required_datasets(config):
        _check_populations(config)
