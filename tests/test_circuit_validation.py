import json
from contextlib import contextmanager
from distutils.dir_util import copy_tree

import h5py

from utils import setup_tempdir

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

import bluepysnap.circuit_validation as test_module
from bluepysnap.circuit_validation import Error, ErrorLevel

TEST_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = TEST_DIR / 'data'


@contextmanager
def _copy_circuit():
    """Copies test/data circuit to a temp directory.

    We don't need the whole circuit every time but considering this is a copy into a temp dir,
    it should be fine.
    Returns:
        yields a path to the copy of the config file
    """
    with setup_tempdir() as tmp_dir:
        copy_tree(str(TEST_DATA_DIR), tmp_dir)
        circuit_copy_path = Path(tmp_dir)
        yield (circuit_copy_path, circuit_copy_path / 'circuit_config.json')


@contextmanager
def _edit_config(config_path):
    """Context manager within which you can edit a circuit config. Edits are saved on the context
    manager leave.

    Args:
        config_path (Path): path to config

    Returns:
        Yields a json dict instance of the config_path. This instance will be saved as the config.
    """
    with config_path.open('r') as f:
        config = json.load(f)
    try:
        yield config
    finally:
        with config_path.open('w') as f:
            json.dump(config, f)


def test_correct_circuit():
    errors = test_module.validate(str(TEST_DATA_DIR / 'circuit_config.json'))
    assert errors == []


def test_no_config_components():
    with _copy_circuit() as (_, config_copy_path):
        with _edit_config(config_copy_path) as config:
            del config['components']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(ErrorLevel.FATAL, 'No "components" in config')]


def test_no_config_networks():
    with _copy_circuit() as (_, config_copy_path):
        with _edit_config(config_copy_path) as config:
            del config['networks']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(ErrorLevel.FATAL, 'No "networks" in config')]


def test_no_config_nodes():
    with _copy_circuit() as (_, config_copy_path):
        with _edit_config(config_copy_path) as config:
            del config['networks']['nodes']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(ErrorLevel.FATAL, 'No "nodes" in config "networks"')]


def test_no_config_edges():
    with _copy_circuit() as (_, config_copy_path):
        with _edit_config(config_copy_path) as config:
            del config['networks']['edges']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(ErrorLevel.FATAL, 'No "edges" in config "networks"')]


def test_invalid_config_nodes_file():
    with _copy_circuit() as (_, config_copy_path):
        with _edit_config(config_copy_path) as config:
            del config['networks']['nodes'][0]['nodes_file']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(ErrorLevel.FATAL, 'Invalid "nodes_file": None')]

        with _edit_config(config_copy_path) as config:
            config['networks']['nodes'][0]['nodes_file'] = '/'
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(ErrorLevel.FATAL, 'Invalid "nodes_file": /')]


def test_invalid_config_nodes_type_file():
    with _copy_circuit() as (_, config_copy_path):
        with _edit_config(config_copy_path) as config:
            config['networks']['nodes'][0]['node_types_file'] = '/'
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(ErrorLevel.FATAL, 'Invalid "node_types_file": /')]


def test_invalid_config_edges_file():
    with _copy_circuit() as (_, config_copy_path):
        with _edit_config(config_copy_path) as config:
            del config['networks']['edges'][0]['edges_file']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(ErrorLevel.FATAL, 'Invalid "edges_file": None')]

        with _edit_config(config_copy_path) as config:
            config['networks']['edges'][0]['edges_file'] = '/'
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(ErrorLevel.FATAL, 'Invalid "edges_file": /')]


def test_invalid_config_edges_type_file():
    with _copy_circuit() as (_, config_copy_path):
        with _edit_config(config_copy_path) as config:
            config['networks']['edges'][0]['edge_types_file'] = '/'
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(ErrorLevel.FATAL, 'Invalid "edge_types_file": /')]


def test_no_nodes_h5():
    with _copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            del h5f['nodes']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [
            Error(ErrorLevel.FATAL, 'No "nodes" in {}.'.format(nodes_file)),
            Error(ErrorLevel.FATAL, 'No node population for "/edges/default/source_node_id"'),
            Error(ErrorLevel.FATAL, 'No node population for "/edges/default/target_node_id"'),
        ]


def test_no_required_node_population_datasets():
    required_datasets = ['node_type_id', 'node_group_id', 'node_group_index']
    for ds in required_datasets:
        with _copy_circuit() as (circuit_copy_path, config_copy_path):
            nodes_file = circuit_copy_path / 'nodes.h5'
            with h5py.File(nodes_file, 'r+') as h5f:
                del h5f['nodes/default/' + ds]
            errors = test_module.validate(str(config_copy_path))
            assert errors == [Error(ErrorLevel.FATAL, 'Population default of {} misses datasets {}'.
                                    format(nodes_file, [ds]))]


def test_no_required_node_group_datasets():
    required_datasets = ['model_template', 'model_type']
    with _copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            for ds in required_datasets:
                del h5f['nodes/default/0/' + ds]
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(ErrorLevel.FATAL,
                                'Group default of {} misses required fields: {}'
                                .format(nodes_file, required_datasets))]


def test_no_required_bio_node_group_datasets():
    required_datasets = sorted(['morphology', 'x', 'y', 'z'])
    with _copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            for ds in required_datasets:
                del h5f['nodes/default/0/' + ds]
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(ErrorLevel.FATAL,
                                'Group default of {} misses biophysical fields: {}'
                                .format(nodes_file, required_datasets))]


def test_no_rotation_bio_node_group_datasets():
    required_datasets = ['rotation_angle_xaxis', 'rotation_angle_yaxis', 'rotation_angle_zaxis']
    with _copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            for ds in required_datasets:
                del h5f['nodes/default/0/' + ds]
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(ErrorLevel.FATAL,
                                'Group default of {} has no rotation fields'.format(nodes_file))]


def test_no_bio_component_dirs():
    dirs = ['morphologies_dir', 'mechanisms_dir', 'biophysical_neuron_models_dir']
    for dir_ in dirs:
        with _copy_circuit() as (_, config_copy_path):
            with _edit_config(config_copy_path) as config:
                del config['components'][dir_]
            errors = test_module.validate(str(config_copy_path))
            # multiplication by 2 because we have 2 populations, each produces the same error.
            assert errors == 2 * [Error(ErrorLevel.FATAL,
                                        'Invalid components "{}": {}'.format(dir_, None))]


def test_no_morph_files():
    with _copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            h5f['nodes/default/0/morphology'][0] = 'noname'
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(
            ErrorLevel.WARNING,
            'missing 1 files in group morphology: default[{}]:\n\tnoname.swc\n'.format(nodes_file))]


def test_no_template_files():
    with _copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            h5f['nodes/default/0/model_template'][0] = 'hoc:noname'
        errors = test_module.validate(str(config_copy_path))
        assert errors == [
            Error(ErrorLevel.WARNING,
                  'missing 1 files in group model_template: default[{}]:\n\tnoname.hoc\n'
                  .format(nodes_file))]


def test_no_edges_h5():
    with _copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            del h5f['edges']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(ErrorLevel.FATAL, 'No "edges" in {}.'.format(edges_file))]


def test_no_required_edge_population_datasets():
    required_datasets = sorted([
        'edge_type_id', 'source_node_id', 'target_node_id', 'edge_group_id', 'edge_group_index'])
    with _copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            for ds in required_datasets:
                del h5f['edges/default/' + ds]
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(ErrorLevel.FATAL, 'Population default of {} misses datasets {}'.
                                format(edges_file, required_datasets))]


def test_no_edge_all_node_ids():
    node_ids_ds = ['node_group_id', 'node_type_id']
    with _copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            for ds in node_ids_ds:
                del h5f['nodes/default/' + ds]
        errors = test_module.validate(str(config_copy_path))
        assert errors == [
            Error(ErrorLevel.FATAL,
                  'Population default of {} misses datasets {}'.format(nodes_file, node_ids_ds)),
            Error(ErrorLevel.FATAL,
                  '/edges/default/source_node_id does not have node ids in its node population'),
            Error(ErrorLevel.FATAL,
                  '/edges/default/target_node_id does not have node ids in its node population')]


def test_invalid_edge_node_ids():
    with _copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            h5f['edges/default/source_node_id'][0] = 99999
            h5f['edges/default/target_node_id'][0] = 99999
        errors = test_module.validate(str(config_copy_path))
        assert errors == [
            Error(ErrorLevel.FATAL,
                  '/edges/default/source_node_id misses node ids in its node population: [99999]'),
            Error(ErrorLevel.FATAL,
                  '/edges/default/target_node_id misses node ids in its node population: [99999]'),
            Error(ErrorLevel.FATAL, 'Population {} edges [99999] have node ids [0 1] instead of '
                                    'single id 2'.format(edges_file)),
            Error(ErrorLevel.FATAL, 'Population {} edges [99999] have node ids [0 1] instead of '
                                    'single id 0'.format(edges_file)),
        ]
