import json
import tempfile

import pytest
from libsonata._libsonata import SonataError

import bluepysnap.circuit as test_module
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.sonata_constants import Node

from utils import copy_config, copy_test_data, edit_config


@pytest.mark.parametrize(
    "config",
    [
        {"metadata": {"status": "partial"}},
        {
            "metadata": {"status": "partial"},
            "components": {
                "morphologies_dir": "some/morph/dir",
            },
            "networks": {
                "nodes": [],
            },
        },
        {
            "metadata": {"status": "partial"},
            "components": {
                "morphologies_dir": "some/morph/dir",
            },
            "networks": {
                "edges": [],
            },
        },
    ],
)
def test_partial_circuit_config_empty(config):
    with tempfile.NamedTemporaryFile(mode="w+") as config_file:
        config_file.write(json.dumps(config))
        config_file.flush()
        circuit = test_module.Circuit(config_file.name)

    with pytest.raises(BluepySnapError):
        assert circuit.get_node_population_config("default")
    with pytest.raises(BluepySnapError):
        assert circuit.get_edge_population_config("default")

    assert circuit.nodes
    assert circuit.nodes.population_names == []
    # property_values returns empty sets if values is empty
    assert list(circuit.nodes.values()) == []
    with pytest.raises(BluepySnapError):
        circuit.nodes.ids()

    assert circuit.edges
    assert circuit.edges
    assert circuit.edges.population_names == []
    # property_values returns empty sets if values is empty
    assert list(circuit.edges.values()) == []
    with pytest.raises(BluepySnapError):
        circuit.edges.ids()


@pytest.mark.parametrize(
    "nodes_update, edges_update",
    [
        ({"populations": {"default": {}}}, {"populations": {"default": {}}}),
    ],
)
def test_partial_circuit_config_partial(nodes_update, edges_update):
    with copy_test_data("nodes.h5") as (_, nodes_path):
        with copy_test_data("edges.h5") as (_, edges_path):
            with copy_config("circuit_config.json") as config_path:
                with edit_config(config_path) as config:
                    config["networks"]["nodes"][0].update(nodes_update)
                    config["networks"]["nodes"][0]["nodes_file"] = str(nodes_path)
                    config["networks"]["edges"][0].update(edges_update)
                    config["networks"]["edges"][0]["edges_file"] = str(edges_path)

                circuit = test_module.Circuit(config_path)

            assert circuit.get_node_population_config("default")
            assert circuit.nodes
            assert circuit.nodes["default"].type
            assert circuit.nodes["default"].size
            assert circuit.nodes["default"].source_in_edges()
            assert circuit.nodes["default"].target_in_edges()
            assert circuit.nodes["default"].config
            assert circuit.nodes["default"].property_names
            assert circuit.nodes["default"].container_property_names(Node)
            assert circuit.nodes["default"].property_values("layer")
            assert circuit.nodes["default"].property_dtypes is not None
            assert circuit.nodes["default"].ids() is not None
            assert circuit.nodes["default"].get() is not None
            assert circuit.nodes["default"].positions() is not None
            assert circuit.nodes["default"].orientations() is not None
            assert circuit.nodes["default"].count()
            assert circuit.nodes["default"].morph
            assert circuit.nodes["default"].models
            assert circuit.nodes["default"].h5_filepath
            assert circuit.nodes["default"]._properties.spatial_segment_index_dir == ""

            assert circuit.nodes.population_names == ["default"]
            assert list(circuit.nodes.values())

            assert [item.id for item in circuit.nodes.ids()] == [0, 1, 2]

            assert circuit.get_edge_population_config("default")
            assert circuit.edges
            assert circuit.edges["default"].size
            assert circuit.edges["default"].type
            assert circuit.edges["default"].source
            assert circuit.edges["default"].target
            assert circuit.edges["default"].config
            assert circuit.edges["default"].property_names
            assert circuit.edges["default"].property_dtypes is not None
            assert circuit.edges["default"].container_property_names(Node) == []
            assert circuit.edges["default"].ids() is not None
            assert circuit.edges["default"].get([1, 2], None) is not None
            assert circuit.edges["default"].positions([1, 2], "afferent", "center") is not None
            assert circuit.edges["default"].afferent_nodes(None) is not None
            assert circuit.edges["default"].efferent_nodes(None) is not None
            assert circuit.edges["default"].pathway_edges(0) is not None
            assert circuit.edges["default"].afferent_edges(0) is not None
            assert circuit.edges["default"].efferent_edges(0) is not None
            assert circuit.edges["default"].iter_connections() is not None
            assert circuit.edges["default"].h5_filepath
            assert circuit.nodes["default"]._properties.spatial_segment_index_dir == ""

            assert circuit.edges.population_names == ["default"]
            assert list(circuit.edges.values())

            assert [item.id for item in circuit.edges.ids()] == [0, 1, 2, 3]
