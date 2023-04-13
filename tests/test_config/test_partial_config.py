import json
import tempfile

import pytest
from libsonata._libsonata import SonataError

import bluepysnap.circuit as test_module
from bluepysnap.exceptions import BluepySnapError

from utils import copy_config, edit_config, copy_test_data


def test_partial_circuit_config_wrong_status():
    with copy_config("circuit_config.json") as config_path:
        with edit_config(config_path) as config:
            config["metadata"] = {"status": "NOT A TYPE"}

        with pytest.raises(SonataError):
            test_module.Circuit(config_path)


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
            assert circuit.nodes["default"].type == "biophysical"
            assert circuit.nodes.population_names == ["default"]
            assert list(circuit.nodes.values())

            assert [item.id for item in circuit.nodes.ids()] == [0, 1, 2]

            assert circuit.get_edge_population_config("default")
            assert circuit.edges
            assert circuit.edges["default"].type == "chemical"
            assert circuit.edges.population_names == ["default"]
            assert list(circuit.edges.values())

            assert [item.id for item in circuit.edges.ids()] == [0, 1, 2, 3]


def test_partial_circuit_config_full():
    with copy_config("circuit_config.json") as config_path:
        with edit_config(config_path) as config:
            config["metadata"] = {"status": "partial"}

        circuit = test_module.Circuit(config_path)

    assert circuit.get_node_population_config("default")
    assert circuit.get_edge_population_config("default")
    assert circuit.nodes.population_names
    assert circuit.edges.population_names
