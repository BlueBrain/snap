try:
    from unittest.mock import patch
except ImportError:
    from mock import patch
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path
import h5py

import six
import bluepysnap.circuit_validation as test_module
from bluepysnap.circuit_validation import Error, BbpError
import numpy as np

from utils import TEST_DATA_DIR, copy_circuit, edit_config


def test_error_comparison():
    err = Error(Error.WARNING, 'hello')
    # we don't use `err == 'hello'` because of py27 compatibility
    assert (err == 'hello') is False


def test_ok_circuit():
    errors = test_module.validate(str(TEST_DATA_DIR / 'circuit_config.json'))
    assert errors == []


def test_no_config_components():
    with copy_circuit() as (_, config_copy_path):
        with edit_config(config_copy_path) as config:
            del config['components']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'No "components" in config')]


def test_no_config_networks():
    with copy_circuit() as (_, config_copy_path):
        with edit_config(config_copy_path) as config:
            del config['networks']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'No "networks" in config')]


def test_no_config_nodes():
    with copy_circuit() as (_, config_copy_path):
        with edit_config(config_copy_path) as config:
            del config['networks']['nodes']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'No "nodes" in config "networks"')]


def test_no_config_edges():
    with copy_circuit() as (_, config_copy_path):
        with edit_config(config_copy_path) as config:
            del config['networks']['edges']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'No "edges" in config "networks"')]


def test_invalid_config_nodes_file():
    with copy_circuit() as (_, config_copy_path):
        with edit_config(config_copy_path) as config:
            del config['networks']['nodes'][0]['nodes_file']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'Invalid "nodes_file": None')]

        with edit_config(config_copy_path) as config:
            config['networks']['nodes'][0]['nodes_file'] = '/'
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'Invalid "nodes_file": /')]


def test_invalid_config_nodes_type_file():
    with copy_circuit() as (_, config_copy_path):
        with edit_config(config_copy_path) as config:
            config['networks']['nodes'][0]['node_types_file'] = '/'
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'Invalid "node_types_file": /')]


def test_invalid_config_edges_file():
    with copy_circuit() as (_, config_copy_path):
        with edit_config(config_copy_path) as config:
            del config['networks']['edges'][0]['edges_file']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'Invalid "edges_file": None')]

        with edit_config(config_copy_path) as config:
            config['networks']['edges'][0]['edges_file'] = '/'
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'Invalid "edges_file": /')]


def test_invalid_config_edge_types_file():
    with copy_circuit() as (_, config_copy_path):
        with edit_config(config_copy_path) as config:
            config['networks']['edges'][0]['edge_types_file'] = '/'
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'Invalid "edge_types_file": /')]


def test_no_nodes_h5():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            del h5f['nodes']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [
            Error(Error.FATAL, 'No "nodes" in {}.'.format(nodes_file)),
            Error(Error.FATAL, 'No node population for "/edges/default/source_node_id"'),
            Error(Error.FATAL, 'No node population for "/edges/default/target_node_id"'),
        ]


def test_ok_node_ids_dataset():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            h5f['nodes/default/node_id'] = list(range(len(h5f['nodes/default/node_type_id'])))
        errors = test_module.validate(str(config_copy_path))
        assert errors == []


def test_no_required_node_single_population_datasets():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            del h5f['nodes/default2/']
            del h5f['nodes/default/node_group_id']
            del h5f['nodes/default/node_group_index']
        errors = test_module.validate(str(config_copy_path))
        assert errors == []
        with h5py.File(nodes_file, 'r+') as h5f:
            del h5f['nodes/default/node_type_id']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'Population default of {} misses datasets {}'.
                                format(nodes_file, ['node_type_id']))]


def test_no_required_node_multi_group_datasets():
    required_datasets = ['node_group_id', 'node_group_index']
    for ds in required_datasets:
        with copy_circuit() as (circuit_copy_path, config_copy_path):
            nodes_file = circuit_copy_path / 'nodes.h5'
            with h5py.File(nodes_file, 'r+') as h5f:
                del h5f['nodes/default/' + ds]
                h5f.copy('nodes/default/0', 'nodes/default/1')
            errors = test_module.validate(str(config_copy_path))
            assert errors == [Error(Error.FATAL, 'Population default of {} misses datasets {}'.
                                    format(nodes_file, [ds]))]


def test_no_required_node_group_datasets():
    required_datasets = ['model_template', 'model_type']
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            for ds in required_datasets:
                del h5f['nodes/default/0/' + ds]
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL,
                                'Group default/0 of {} misses required fields: {}'
                                .format(nodes_file, required_datasets))]


def test_ok_nonbio_node_group_datasets():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            h5f['nodes/default/0/model_type'][:] = ''
        errors = test_module.validate(str(config_copy_path))
        assert errors == []


def test_no_required_bio_node_group_datasets():
    required_datasets = sorted(['morphology', 'x', 'y', 'z'])
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            for ds in required_datasets:
                del h5f['nodes/default/0/' + ds]
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL,
                                'Group default/0 of {} misses biophysical fields: {}'
                                .format(nodes_file, required_datasets))]


def test_ok_bio_model_type_in_library():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            data = h5f['nodes/default/0/model_type'][:]
            del h5f['nodes/default/0/model_type']
            h5f.create_dataset('nodes/default/0/model_type', data=np.zeros_like(data, dtype=int))
            dt = h5py.special_dtype(vlen=six.text_type)
            h5f.create_dataset('nodes/default/0/@library/model_type',
                               data=np.array(["biophysical", ], dtype=object), dtype=dt)
        errors = test_module.validate(str(config_copy_path))
        assert errors == []


def test_no_rotation_bio_node_group_datasets():
    angle_datasets = ['rotation_angle_xaxis', 'rotation_angle_yaxis', 'rotation_angle_zaxis']
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            for ds in angle_datasets:
                del h5f['nodes/default/0/' + ds]
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.WARNING,
                                'Group default/0 of {} has no rotation fields'.format(nodes_file))]


def test_no_rotation_bbp_node_group_datasets():
    angle_datasets = ['rotation_angle_xaxis', 'rotation_angle_yaxis', 'rotation_angle_zaxis']
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            for ds in angle_datasets:
                del h5f['nodes/default/0/' + ds]
            h5f['nodes/default/0/orientation_w'] = 0
        errors = test_module.validate(str(config_copy_path), bbp_check=True)
        assert errors == [
            Error(Error.WARNING, 'Group default/0 of {} has no rotation fields'.format(nodes_file)),
            BbpError(Error.WARNING,
                     'Group default/0 of {} has no rotation fields'.format(nodes_file))
        ]


def test_no_bio_component_dirs():
    dirs = ['morphologies_dir', 'biophysical_neuron_models_dir']
    for dir_ in dirs:
        with copy_circuit() as (_, config_copy_path):
            with edit_config(config_copy_path) as config:
                del config['components'][dir_]
            errors = test_module.validate(str(config_copy_path), True)
            # multiplication by 2 because we have 2 populations, each produces the same error.
            assert errors == 2 * [BbpError(Error.FATAL,
                                           'Invalid components "{}": {}'.format(dir_, None))]


@patch('bluepysnap.circuit_validation.MAX_MISSING_FILES_DISPLAY', 1)
def test_no_morph_files():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            h5f['nodes/default/0/morphology'][0] = 'noname'
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(
            Error.WARNING,
            'missing 1 files in group morphology: default/0[{}]:\n\tnoname.swc\n'.format(
                nodes_file))]

        with h5py.File(nodes_file, 'r+') as h5f:
            morph = h5f['nodes/default/0/morphology']
            morph[:] = ['noname' + str(i) for i in range(len(morph))]
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(
            Error.WARNING,
            'missing 3 files in group morphology: default/0[{}]:\n\tnoname0.swc\n\t...\n'.format(
                nodes_file))]


@patch('bluepysnap.circuit_validation.MAX_MISSING_FILES_DISPLAY', 1)
def test_no_morph_library_files():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            grp = h5f['nodes/default/0']
            str_dtype = h5py.special_dtype(vlen=str)
            grp.create_dataset('@library/morphology', shape=(1,), dtype=str_dtype)
            grp['@library/morphology'][:] = u'noname'
            shape = grp['morphology'].shape
            del grp['morphology']
            grp.create_dataset('morphology', shape=shape, fillvalue=0)
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(
            Error.WARNING,
            'missing 1 files in group morphology: default/0[{}]:\n\tnoname.swc\n'.format(
                nodes_file))]


def test_no_template_files():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            h5f['nodes/default/0/model_template'][0] = 'hoc:noname'
        errors = test_module.validate(str(config_copy_path))
        assert errors == [
            Error(Error.WARNING,
                  'missing 1 files in group model_template: default/0[{}]:\n\tnoname.hoc\n'
                  .format(nodes_file))]


@patch('bluepysnap.circuit_validation.MAX_MISSING_FILES_DISPLAY', 1)
def test_no_template_library_files():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            grp = h5f['nodes/default/0']
            str_dtype = h5py.special_dtype(vlen=str)
            grp.create_dataset('@library/model_template', shape=(1,), dtype=str_dtype)
            grp['@library/model_template'][:] = u'hoc:noname'
            shape = grp['model_template'].shape
            del grp['model_template']
            grp.create_dataset('model_template', shape=shape, fillvalue=0)
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(
            Error.WARNING,
            'missing 1 files in group model_template: default/0[{}]:\n\tnoname.hoc\n'.format(
                nodes_file))]


def test_no_edges_h5():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            del h5f['edges']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'No "edges" in {}.'.format(edges_file))]


def test_no_edge_group():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            del h5f['edges/default/0']
        errors = test_module.validate(str(config_copy_path))
        assert errors == []


def test_no_edge_group_missing_requiered_datasets():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        required_datasets = sorted([
            'edge_type_id', 'source_node_id', 'target_node_id'])
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            del h5f['edges/default/0']
            for ds in required_datasets:
                del h5f['edges/default/' + ds]
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'Population default of {} misses datasets {}'.
                                format(edges_file, required_datasets))]


def test_no_edge_group_no_optional_datasets():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        optional_datasets = sorted(['edge_group_id', 'edge_group_index'])
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            del h5f['edges/default/0']
            for ds in optional_datasets:
                del h5f['edges/default/' + ds]
        errors = test_module.validate(str(config_copy_path))
        assert errors == []


def test_no_required_edge_population_datasets_one_group():
    required_datasets = sorted([
        'edge_type_id', 'source_node_id', 'target_node_id'])
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            for ds in required_datasets:
                del h5f['edges/default/' + ds]
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'Population default of {} misses datasets {}'.
                                format(edges_file, required_datasets))]


def test_missing_optional_edge_population_datasets_one_group():
    optional_datasets = sorted(['edge_group_id', 'edge_group_index'])
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            for ds in optional_datasets:
                del h5f['edges/default/' + ds]
        errors = test_module.validate(str(config_copy_path))
        assert errors == []


def test_no_required_edge_population_datasets_multiple_groups():
    required_datasets = sorted([
        'edge_type_id', 'source_node_id', 'target_node_id', 'edge_group_id', 'edge_group_index'])
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            for ds in required_datasets:
                del h5f['edges/default/' + ds]
            h5f.create_group('edges/default/1')
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'Population default of {} misses datasets {}'.
                                format(edges_file, required_datasets))]


def test_edge_population_multiple_groups():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            h5f.create_group('edges/default/1')
        errors = test_module.validate(str(config_copy_path), bbp_check=True)
        assert BbpError(Error.WARNING, 'Population default of {} have multiple groups. '
                                       'Cannot be read via bluepysnap or libsonata'.
                        format(edges_file)) in errors


def test_edge_population_missing_edge_group_id_one_group():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            del h5f['edges/default/edge_group_id']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'Population default of {} misses dataset {}'.
                                format(edges_file, {"edge_group_id"}))]


def test_edge_population_missing_edge_group_index_one_group():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            del h5f['edges/default/edge_group_index']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'Population default of {} misses dataset {}'.
                                format(edges_file, {"edge_group_index"}))]


def test_edge_population_missing_edge_group_id_index_one_group():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            del h5f['edges/default/edge_group_index']
            del h5f['edges/default/edge_group_id']
        errors = test_module.validate(str(config_copy_path))
        assert errors == []


def test_edge_population_edge_group_different_length():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            del h5f['edges/default/edge_group_index']
            h5f.create_dataset('edges/default/edge_group_index', data=[0, 1, 2, 3, 4])
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL,
                                'Population default of {} "edge_group_id" and "edge_group_index" of different sizes'.
                                format(edges_file))]


def test_edge_population_wrong_group_id():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            del h5f['edges/default/edge_group_id']
            h5f.create_dataset('edges/default/edge_group_id', data=[0, 1, 0, 0])
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'Population default of {} misses group(s): {}'.
                                format(edges_file, {1}))]


def test_edge_population_ok_group_index():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            del h5f['edges/default/edge_group_id']
            del h5f['edges/default/edge_group_index']
            h5f.create_group('edges/default/1')
            h5f.create_dataset('edges/default/1/test', data=[1, 1])
            h5f.create_dataset('edges/default/edge_group_id', data=[0, 1, 0, 1])
            h5f.create_dataset('edges/default/edge_group_index', data=[0, 0, 1, 1])
        errors = test_module.validate(str(config_copy_path))
        assert errors == []


def test_edge_population_wrong_group_index():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            del h5f['edges/default/edge_group_index']
            h5f.create_dataset('edges/default/edge_group_index', data=[0, 1, 2, 12])
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'Group default/0 in file {} should have ids up to {}'.
                                format(edges_file, 12))]


def test_edge_population_wrong_group_index_multi_group():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            del h5f['edges/default/edge_group_id']
            del h5f['edges/default/edge_group_index']
            h5f.create_group('edges/default/1')
            h5f.create_dataset('edges/default/1/test', data=[1, 1])
            h5f.create_dataset('edges/default/edge_group_id', data=[0, 1, 0, 1])
            h5f.create_dataset('edges/default/edge_group_index', data=[0, 0, 12, 1])
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'Group default/0 in file {} should have ids up to {}'.
                                format(edges_file, 12))]


def test_no_required_bbp_edge_group_datasets():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            del h5f['edges/default/0/syn_weight']
        errors = test_module.validate(str(config_copy_path), True)
        assert errors == [BbpError(Error.WARNING, 'Group default/0 of {} misses fields: {}'.
                                   format(edges_file, ['syn_weight']))]


def test_no_edge_source_to_target():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            del h5f['edges/default/indices/source_to_target']
            del h5f['edges/default/indices/target_to_source']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [Error(Error.FATAL, 'No "source_to_target" in {}'.format(edges_file)),
                          Error(Error.FATAL, 'No "target_to_source" in {}'.format(edges_file))]


def test_no_edge_all_node_ids():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / 'nodes.h5'
        with h5py.File(nodes_file, 'r+') as h5f:
            del h5f['nodes/default/0']
        errors = test_module.validate(str(config_copy_path))
        assert errors == [
            Error(Error.FATAL,
                  '/edges/default/source_node_id does not have node ids in its node population'),
            Error(Error.FATAL,
                  '/edges/default/target_node_id does not have node ids in its node population')]


def test_invalid_edge_node_ids():
    with copy_circuit() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / 'edges.h5'
        with h5py.File(edges_file, 'r+') as h5f:
            h5f['edges/default/source_node_id'][0] = 99999
            h5f['edges/default/target_node_id'][0] = 99999
        errors = test_module.validate(str(config_copy_path))
        assert errors == [
            Error(Error.FATAL,
                  '/edges/default/source_node_id misses node ids in its node population: [99999]'),
            Error(Error.FATAL,
                  '/edges/default/target_node_id misses node ids in its node population: [99999]'),
            Error(Error.FATAL, 'Population {} edges [99999] have node ids [0 1] instead of '
                               'single id 2'.format(edges_file)),
            Error(Error.FATAL, 'Population {} edges [99999] have node ids [0 1] instead of '
                               'single id 0'.format(edges_file)),
        ]
