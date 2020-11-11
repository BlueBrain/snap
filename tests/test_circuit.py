import json

import pytest


from bluepysnap.nodes import NodePopulation, Nodes
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
                'node_types_file': str(TEST_DATA_DIR / 'node_types.csv'),
            }
    )
    assert isinstance(circuit.nodes, Nodes)
    assert isinstance(circuit.edges, dict)
    assert list(circuit.edges) == ['default']
    assert isinstance(circuit.edges['default'], EdgePopulation)
    assert sorted(list(circuit.nodes)) == ['default', 'default2']
    assert isinstance(circuit.nodes['default'], NodePopulation)
    assert isinstance(circuit.nodes['default2'], NodePopulation)
    assert (sorted(circuit.node_sets) ==
            sorted(json.load(open(str(TEST_DATA_DIR / 'node_sets.json')))))


def test_duplicate_node_populations():
    circuit = test_module.Circuit(
        str(TEST_DATA_DIR / 'circuit_config_duplicate.json')
    )
    with pytest.raises(BluepySnapError):
        circuit.nodes


def test_duplicate_edge_populations():
    circuit = test_module.Circuit(
        str(TEST_DATA_DIR / 'circuit_config_duplicate.json')
    )
    with pytest.raises(BluepySnapError):
        circuit.edges


def test_no_node_set():
    circuit = test_module.Circuit(
        str(TEST_DATA_DIR / 'circuit_config_duplicate.json')
    )
    # replace the _config dict with random one that does not contain "node_sets_file" key
    circuit._config = {"key": "value"}
    assert circuit.node_sets == {}
