import pytest
import h5py

from bluepysnap.nodes import NodePopulation
from bluepysnap.edges import EdgePopulation
from bluepysnap.exceptions import BluepySnapError

import bluepysnap.circuit as test_module
from utils import TEST_DATA_DIR


def test_all():
    circuit = test_module.Circuit(
        str(TEST_DATA_DIR / 'circuit_config.json'))
    assert (
            circuit.config['networks']['nodes'][0] ==
            {
                'nodes_file': str(TEST_DATA_DIR / 'nodes.h5'),
                'node_types_file': None,
            }
    )
    assert isinstance(circuit.nodes, dict)
    assert isinstance(circuit.edges, dict)
    assert list(circuit.edges) == ['default']
    assert isinstance(circuit.edges['default'], EdgePopulation)
    assert sorted(list(circuit.nodes)) == ['default', 'default2']
    assert isinstance(circuit.nodes['default'], NodePopulation)
    assert isinstance(circuit.nodes['default2'], NodePopulation)


def test_duplicate_population():
    circuit = test_module.Circuit(
        str(TEST_DATA_DIR / 'circuit_config_duplicate.json')
    )
    with pytest.raises(BluepySnapError):
        circuit.nodes


def test_close_contexts():
    with test_module.Circuit(str(TEST_DATA_DIR / 'circuit_config.json')) as circuit:
        edge = circuit.edges['default']
        edge.size
        node = circuit.nodes['default']
        node.size
        node_file = circuit.config['networks']['nodes'][0]['nodes_file']
        edge_file = circuit.config['networks']['edges'][0]['edges_file']

    with h5py.File(node_file, "r+") as h5:
        list(h5)

    with h5py.File(edge_file, "r+") as h5:
        list(h5)

def test_close_contexts_errors_extracted_obj():
    with test_module.Circuit(str(TEST_DATA_DIR / 'circuit_config.json')) as circuit:
        edges = circuit.edges['default']
        edges.size
        nodes = circuit.nodes['default']
        nodes.size

    with pytest.raises(BluepySnapError):
        circuit.nodes

    with pytest.raises(BluepySnapError):
        circuit.edges

    with pytest.raises(BluepySnapError):
        nodes.property_names

    with pytest.raises(BluepySnapError):
        edges.property_names
