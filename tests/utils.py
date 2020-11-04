"""Module providing utility functions for the tests"""

import shutil
import tempfile
import json
import six
import mock
from contextlib import contextmanager
from distutils.dir_util import copy_tree

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

from bluepysnap.nodes import Nodes, NodeStorage


TEST_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = TEST_DIR / 'data'


@contextmanager
def setup_tempdir(cleanup=True):
    temp_dir = str(Path(tempfile.mkdtemp()).resolve())
    try:
        yield temp_dir
    finally:
        if cleanup:
            shutil.rmtree(temp_dir)


@contextmanager
def copy_circuit(config='circuit_config.json'):
    """Copies test/data circuit to a temp directory.

    We don't need the whole circuit every time but considering this is a copy into a temp dir,
    it should be fine.
    Returns:
        yields a path to the copy of the config file
    """
    with setup_tempdir() as tmp_dir:
        copy_tree(str(TEST_DATA_DIR), tmp_dir)
        circuit_copy_path = Path(tmp_dir)
        yield circuit_copy_path, circuit_copy_path / config


@contextmanager
def copy_config(config='circuit_config.json'):
    """Copies config to a temp directory.

    Returns:
        yields a path to the copy of the config file
    """
    with setup_tempdir() as tmp_dir:
        output = Path(tmp_dir, config)
        shutil.copy(str(TEST_DATA_DIR / config), str(output))
        yield output


@contextmanager
def edit_config(config_path):
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
            f.write(six.u(json.dumps(config)))


def create_node_population(filepath, pop_name, circuit=None, node_sets=None):
    """Creates a node population.
    Args:
        filepath (str): path to the node file.
        pop_name (str): population name inside the file.
        circuit (Mock/Circuit): either a real circuit or a Mock containing the nodes.
        node_sets: (Mock/NodeSets): either a real node_sets or a mocked node_sets.
    Returns:
        NodePopulation: return a node population.
    """
    config = {
        'nodes_file': filepath,
        'node_types_file': None,
    }
    if circuit is None:
        circuit = mock.Mock()
    if node_sets is not None:
        circuit.node_sets = node_sets
    node_pop = NodeStorage(config, circuit).population(pop_name)
    circuit.config = {"networks": {"nodes": [config]}}
    circuit.nodes = Nodes(circuit)
    return node_pop
