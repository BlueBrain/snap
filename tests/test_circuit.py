import os

import pytest

from bluepysnap.nodes import NodePopulation
from bluepysnap.edges import EdgePopulation
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
    assert list(circuit.edges) == ['default']
    assert isinstance(circuit.edges['default'], EdgePopulation)
    assert sorted(list(circuit.nodes)) == ['default', 'default2']
    assert isinstance(circuit.nodes['default'], NodePopulation)
    assert isinstance(circuit.nodes['default2'], NodePopulation)


def test_duplicate_population():
    circuit = test_module.Circuit(
        os.path.join(TEST_DATA_DIR, 'circuit_config_duplicate.json')
    )
    with pytest.raises(BluepySnapError):
        circuit.nodes

