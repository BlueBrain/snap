import json
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from distutils.dir_util import copy_tree

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
    with TemporaryDirectory() as tmp_dir:
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
