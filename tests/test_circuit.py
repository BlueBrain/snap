import os

import pytest

from bluepysnap.nodes import NodePopulation
from bluepysnap.edges import EdgePopulation
from bluepysnap.exceptions import BlueSnapError

import bluepysnap.circuit as test_module


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, 'data')


def test_all():
    circuit = test_module.Circuit(
        os.path.join(TEST_DATA_DIR, 'circuit_config.json'),
        node_population='default'
    )
    assert(
        circuit.config['networks']['nodes'][0] ==
        {
            'nodes_file': os.path.join(TEST_DATA_DIR, 'nodes.h5'),
            'node_types_file': None,
        }
    )
    assert isinstance(circuit.nodes, NodePopulation)
    assert isinstance(circuit.edges, dict)
    assert list(circuit.edges) == ['default']
    assert isinstance(circuit.edges['default'], EdgePopulation)


def test_no_population():
    circuit = test_module.Circuit(
        os.path.join(TEST_DATA_DIR, 'circuit_config.json'),
        node_population='no-such-population'
    )
    with pytest.raises(BlueSnapError):
        circuit.nodes


def test_duplicate_population():
    circuit = test_module.Circuit(
        os.path.join(TEST_DATA_DIR, 'circuit_config_duplicate.json')
    )
    with pytest.raises(BlueSnapError):
        circuit.nodes
