import os

import pytest

from bluepysnap.nodes import NodePopulation, NodeStorage
from bluepysnap.edges import EdgePopulation, EdgeStorage
from bluepysnap.exceptions import BluepySnapError

import bluepysnap.circuit as test_module


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, 'data')


def test_all():
    circuit = test_module.Circuit(
        os.path.join(TEST_DATA_DIR, 'circuit_config.json'))
    assert(
        circuit.config['networks']['nodes'][0] ==
        {
            'nodes_file': os.path.join(TEST_DATA_DIR, 'nodes.h5'),
            'node_types_file': None,
        }
    )
    assert isinstance(circuit.nodes, dict)
    assert isinstance(circuit.edges, dict)
    assert circuit.edge_populations == ['default']
    assert isinstance(circuit.edges['default'], EdgePopulation)
    assert sorted(circuit.node_populations) == ['default', 'default2']
    assert isinstance(circuit.nodes['default'], NodePopulation)
    assert isinstance(circuit.nodes['default2'], NodePopulation)

def test_no_population():
    circuit = test_module.Circuit(
        os.path.join(TEST_DATA_DIR, 'circuit_config.json'),
        node_populations='no-such-population'
    )
    with pytest.raises(BluepySnapError):
        circuit.nodes


def test_missing_population():
    circuit = test_module.Circuit(
        os.path.join(TEST_DATA_DIR, 'circuit_config.json'),
        node_populations=['default', 'no-such-population']
    )
    with pytest.raises(BluepySnapError):
        circuit.nodes


def test_duplicate_population():
    circuit = test_module.Circuit(
        os.path.join(TEST_DATA_DIR, 'circuit_config_duplicate.json')
    )
    with pytest.raises(BluepySnapError):
        circuit.nodes

