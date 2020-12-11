import json

import pytest


from bluepysnap.nodes import NodePopulation, Nodes
from bluepysnap.edges import EdgePopulation, Edges
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
    assert isinstance(circuit.edges, Edges)
    assert list(circuit.edges) == ['default', 'default2']
    assert isinstance(circuit.edges['default'], EdgePopulation)
    assert isinstance(circuit.edges['default2'], EdgePopulation)
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
        list(circuit.nodes)


def test_duplicate_edge_populations():
    circuit = test_module.Circuit(
        str(TEST_DATA_DIR / 'circuit_config_duplicate.json')
    )
    with pytest.raises(BluepySnapError):
        list(circuit.edges)


def test_no_node_set():
    circuit = test_module.Circuit(
        str(TEST_DATA_DIR / 'circuit_config_duplicate.json')
    )
    # replace the _config dict with random one that does not contain "node_sets_file" key
    circuit._config = {"key": "value"}
    assert circuit.node_sets == {}


def test_integration():
    circuit = test_module.Circuit(str(TEST_DATA_DIR / 'circuit_config.json'))
    node_ids = circuit.nodes.ids({"mtype": ["L6_Y", "L2_X"]})
    edge_ids = circuit.edges.afferent_edges(node_ids)
    edge_props = circuit.edges.properties(edge_ids, properties=["syn_weight", "delay"])
    edge_reduced = edge_ids.limit(2)
    edge_props_reduced = edge_props.loc[edge_reduced]
    assert edge_props_reduced["syn_weight"].tolist() == [1, 1]
