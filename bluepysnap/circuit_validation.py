"""
Standalone module that validates Sonata circuit. See ``validate`` and ``validate_cli`` functions.
"""
import warnings
from os import path

import click
import h5py

from bluepysnap.config import Config


def _check_required_datasets(config):
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
        if nodes_file is None or not path.isfile(nodes_file):
            warnings.warn('Invalid "nodes_file": {}'.format(nodes_file))
        types_file = nodes_dict.get('node_types_file')
        if types_file is not None and not path.isfile(types_file):
            warnings.warn('Invalid "node_types_file": {}'.format(nodes_file))
    for edges_dict in edges:
        edges_file = edges_dict.get('edges_file')
        if edges_file is None or not path.isfile(edges_file):
            warnings.warn('Invalid "edges_file": {}'.format(edges_file))
        types_file = edges_dict.get('edge_types_file')
        if types_file is not None and not path.isfile(types_file):
            warnings.warn('Invalid "edge_types_file": {}'.format(types_file))
    return True


def _check_components(config):
    components = config.get('components')
    if not components:
        warnings.warn('No "components" in config')
        return
    morphologies = components.get('morphologies_dir')
    if not morphologies or not path.isdir(morphologies):
        warnings.warn('Invalid "morphologies_dir": {}'.format(morphologies))
    mechanisms = components.get('mechanisms_dir')
    if not mechanisms or path.isdir(mechanisms):
        warnings.warn('Invalid "mechanisms_dir": {}'.format(mechanisms))
    biophysics = components.get('biophysical_neuron_models_dir')
    if not biophysics or path.isdir(biophysics):
        warnings.warn('Invalid "biophysical_neuron_models_dir": {}'.format(biophysics))


def _check_node_dynamics_params(params):
    PARAMS_NAMES = ['holding_current', 'threshold_current']
    missing_fields = set(PARAMS_NAMES) - set(params.keys())
    if len(missing_fields) > 0:
        warnings.warn('Dynamics Params {} of {} misses fields: {}'.format(
            params, params.file.filename, missing_fields))


def _check_node_group(group, config):
    GROUP_NAMES = [
        'layer', 'model_template', 'morphology', 'mtype',
        'rotation_angle_xaxis', 'rotation_angle_yaxis', 'rotation_angle_zaxis',
        'x', 'y', 'z', 'dynamics_params']

    missing_fields = set(GROUP_NAMES) - set(group.keys())
    if len(missing_fields) > 0:
        warnings.warn('Group {} of {} misses fields: {}'.format(
            group, group.file.filename, missing_fields))
    if 'dynamics_params' in group:
        _check_node_dynamics_params(group['dynamics_params'])
    morphology_ds = group.get('morphology')
    if morphology_ds:
        for morphology in morphology_ds[:]:
            if not path.isfile(path.join(config['components']['morphologies_dir'], morphology)):
                warnings.warn('non-existent morphology file {} in group {} of {}'.format(
                    morphology, group, group.file.filename
                ))
    model_template_ds = group.get('model_template')
    if model_template_ds:
        for model_template in model_template_ds[:]:
            if not path.isfile(path.join(config['components']['biophysical_neuron_models_dir'],
                    model_template.split('hoc:')[1])):
                warnings.warn('non-existent model_template file {} in group {} of {}'.format(
                    model_template, group, group.file.filename
                ))


def _check_node_population(nodes_dict, config):
    POPULATION_DATASET_NAMES = ['node_type_id', 'node_id', 'node_group_id', 'node_group_index']
    nodes_file = nodes_dict.get('nodes_file')
    with h5py.File(nodes_file) as f:
        assert 'nodes' in f
        for population_name in f['nodes'].keys():
            population_path = '/nodes/' + population_name
            population = f[population_path]
            children_names = population.keys()
            missing_datasets = set(POPULATION_DATASET_NAMES) - set(children_names)
            if len(missing_datasets) > 0:
                warnings.warn('Population {} of {} misses datasets {}'.format(
                    population, nodes_file, missing_datasets))
            for name in children_names:
                if isinstance(population[name], h5py.Group):
                    _check_node_group(population[name], config)


def _check_populations(config):
    networks = config.get('networks')
    nodes = networks.get('nodes')
    # edges = networks.get('edges')
    for nodes_dict in nodes:
        _check_node_population(nodes_dict, config)


def validate(config_file):
    """Validates Sonata circuit

    Args:
        config_file: path to Sonata circuit config file
    """
    config = Config(config_file).resolve()
    _check_components(config)
    if _check_required_datasets(config):
        _check_populations(config)


@click.group()
def cli():
    """The CLI object"""


@cli.command('validate', short_help='Validate Sonata circuit')
@click.argument('config_file', type=click.Path(exists=True, file_okay=True, dir_okay=False))
def validate_cli(config_file):
    """Cli command for validating of Sonata circuit

    Args:
        config_file: path to Sonata circuit config file
    """
    validate(config_file)


if __name__ == '__main__':
    validate('/home/sanin/workspace/snap/tests/data/circuit_config.json')
