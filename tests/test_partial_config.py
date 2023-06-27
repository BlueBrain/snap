import json
import logging
import tempfile

import pytest

import bluepysnap.circuit as test_module
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.sonata_constants import Edge, Node

from utils import TEST_DATA_DIR


def test_partial_circuit_config_minimal():
    config = {
        "metadata": {"status": "partial"},
        "networks": {
            "nodes": [
                {
                    # missing `nodes_file` rises libsonata exception
                    "nodes_file": str(TEST_DATA_DIR / "nodes.h5"),
                    "populations": {
                        "default": {},
                    },
                }
            ],
            "edges": [
                {
                    # missing `edges_file` rises libsonata exception
                    "edges_file": str(TEST_DATA_DIR / "edges.h5"),
                    "populations": {
                        "default": {},
                    },
                }
            ],
        },
    }
    with tempfile.NamedTemporaryFile(mode="w+") as config_file:
        config_file.write(json.dumps(config))
        config_file.flush()
        circuit = test_module.Circuit(config_file.name)

    assert circuit.get_node_population_config("default")
    assert circuit.nodes
    assert circuit.nodes["default"].type == "biophysical"
    assert circuit.nodes["default"].size == 3
    assert circuit.nodes["default"].source_in_edges() == {"default"}
    assert circuit.nodes["default"].target_in_edges() == {"default"}
    assert circuit.nodes["default"].config is not None
    assert circuit.nodes["default"].property_names is not None
    assert circuit.nodes["default"].container_property_names(Node) is not None
    assert circuit.nodes["default"].container_property_names(Edge) == []
    assert circuit.nodes["default"].property_values("layer") == {2, 6}
    assert circuit.nodes["default"].property_dtypes is not None
    assert list(circuit.nodes["default"].ids()) == [0, 1, 2]
    assert circuit.nodes["default"].get() is not None
    assert circuit.nodes["default"].positions() is not None
    assert circuit.nodes["default"].orientations() is not None
    assert circuit.nodes["default"].count() == 3
    assert circuit.nodes["default"].morph is not None
    assert circuit.nodes["default"].models is not None
    assert circuit.nodes["default"].h5_filepath == str(TEST_DATA_DIR / "nodes.h5")
    assert circuit.nodes["default"]._properties.spatial_segment_index_dir == ""

    assert circuit.nodes.population_names == ["default"]
    assert list(circuit.nodes.values())

    assert [item.id for item in circuit.nodes.ids()] == [0, 1, 2]

    assert circuit.get_edge_population_config("default")
    assert circuit.edges
    assert circuit.edges["default"].type == "chemical"
    assert circuit.edges["default"].size == 4
    assert circuit.edges["default"].source is not None
    assert circuit.edges["default"].target is not None
    assert circuit.edges["default"].config is not None
    assert circuit.edges["default"].property_names is not None
    assert circuit.edges["default"].property_dtypes is not None
    assert circuit.edges["default"].container_property_names(Node) == []
    assert circuit.edges["default"].container_property_names(Edge) is not None
    assert list(circuit.edges["default"].ids()) == [0, 1, 2, 3]
    assert list(circuit.edges["default"].get([1, 2], None)) == [1, 2]
    assert circuit.edges["default"].positions([1, 2], "afferent", "center") is not None
    assert list(circuit.edges["default"].afferent_nodes(None)) == [0, 2]
    assert list(circuit.edges["default"].efferent_nodes(None)) == [0, 1]
    assert list(circuit.edges["default"].pathway_edges(0)) == [1, 2]
    assert list(circuit.edges["default"].afferent_edges(0)) == [0]
    assert list(circuit.edges["default"].efferent_edges(0)) == [1, 2]
    assert circuit.edges["default"].iter_connections() is not None
    assert circuit.edges["default"].h5_filepath == str(TEST_DATA_DIR / "edges.h5")
    assert circuit.nodes["default"]._properties.spatial_segment_index_dir == ""

    assert circuit.edges.population_names == ["default"]
    assert list(circuit.edges.values())

    assert [item.id for item in circuit.edges.ids()] == [0, 1, 2, 3]


def test_partial_circuit_config_log(caplog):
    caplog.set_level(logging.INFO)

    config = {"metadata": {"status": "partial"}}

    with tempfile.NamedTemporaryFile(mode="w+") as config_file:
        config_file.write(json.dumps(config))
        config_file.flush()
        test_module.Circuit(config_file.name)

    assert "Loaded PARTIAL circuit config" in caplog.text


def test_partial_circuit_config_empty():
    config = {"metadata": {"status": "partial"}}
    with tempfile.NamedTemporaryFile(mode="w+") as config_file:
        config_file.write(json.dumps(config))
        config_file.flush()
        circuit = test_module.Circuit(config_file.name)

    with pytest.raises(BluepySnapError):
        assert circuit.get_node_population_config("default")
    with pytest.raises(BluepySnapError):
        assert circuit.get_edge_population_config("default")
    with pytest.raises(BluepySnapError):
        circuit.nodes.ids()
    with pytest.raises(BluepySnapError):
        circuit.edges.ids()
