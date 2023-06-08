import json
import pickle

import pandas as pd
import pytest
from libsonata import SonataError

import bluepysnap.circuit as test_module
from bluepysnap.edges import EdgePopulation, Edges
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.nodes import NodePopulation, Nodes

from utils import PICKLED_SIZE_ADJUSTMENT, TEST_DATA_DIR, copy_test_data, edit_config


def test_all():
    circuit = test_module.Circuit(str(TEST_DATA_DIR / "circuit_config.json"))
    assert circuit.config["networks"]["nodes"][0] == {
        "nodes_file": str(TEST_DATA_DIR / "nodes.h5"),
        "populations": {
            "default": {"type": "biophysical"},
            "default2": {"type": "biophysical", "spatial_segment_index_dir": "path/to/node/dir"},
        },
    }
    assert isinstance(circuit.nodes, Nodes)
    assert isinstance(circuit.edges, Edges)
    assert list(circuit.edges) == ["default", "default2"]
    assert isinstance(circuit.edges["default"], EdgePopulation)
    assert isinstance(circuit.edges["default2"], EdgePopulation)
    assert sorted(list(circuit.nodes)) == ["default", "default2"]
    assert isinstance(circuit.nodes["default"], NodePopulation)
    assert isinstance(circuit.nodes["default2"], NodePopulation)
    assert sorted(circuit.node_sets) == sorted(
        json.loads((TEST_DATA_DIR / "node_sets.json").read_text())
    )

    fake_pop = "fake"
    with pytest.raises(
        BluepySnapError, match=f"Population config not found for node population: {fake_pop}"
    ):
        circuit.get_node_population_config(fake_pop)
    with pytest.raises(
        BluepySnapError, match=f"Population config not found for edge population: {fake_pop}"
    ):
        circuit.get_edge_population_config(fake_pop)


def test_duplicate_node_populations():
    with copy_test_data() as (_, config_path):
        with edit_config(config_path) as config:
            config["networks"]["nodes"].append(config["networks"]["nodes"][0])

        match = "Duplicate population|Population default is declared twice"
        with pytest.raises(SonataError, match=match):
            test_module.Circuit(config_path)


def test_duplicate_edge_populations():
    with copy_test_data() as (_, config_path):
        with edit_config(config_path) as config:
            config["networks"]["edges"].append(config["networks"]["edges"][0])

        match = "Duplicate population|Population default is declared twice"
        with pytest.raises(SonataError, match=match):
            test_module.Circuit(config_path)


def test_no_node_set():
    with copy_test_data() as (_, config_path):
        with edit_config(config_path) as config:
            config.pop("node_sets_file")
        circuit = test_module.Circuit(config_path)
        assert circuit.node_sets.content == {}


def test_integration():
    circuit = test_module.Circuit(str(TEST_DATA_DIR / "circuit_config.json"))
    node_ids = circuit.nodes.ids({"mtype": ["L6_Y", "L2_X"]})
    edge_ids = circuit.edges.afferent_edges(node_ids)
    edge_props = circuit.edges.get(edge_ids, properties=["syn_weight", "delay"])
    edge_reduced = edge_ids.limit(2)
    edge_props = pd.concat(df for _, df in edge_props)
    edge_props_reduced = edge_props.loc[edge_reduced]
    assert edge_props_reduced["syn_weight"].tolist() == [1, 1]


def test_pickle(tmp_path):
    circuit = test_module.Circuit(TEST_DATA_DIR / "circuit_config.json")

    pickle_path = tmp_path / "pickle.pkl"
    with open(pickle_path, "wb") as fd:
        pickle.dump(circuit, fd)

    with open(pickle_path, "rb") as fd:
        circuit = pickle.load(fd)

    assert pickle_path.stat().st_size < 60 + PICKLED_SIZE_ADJUSTMENT
    assert list(circuit.edges) == ["default", "default2"]


def test_empty_manifest(tmp_path):
    path = tmp_path / "circuit_config.json"
    with open(path, "w") as fd:
        json.dump(
            {
                "manifest": {},
                "networks": {
                    "nodes": {},
                    "edges": {},
                },
            },
            fd,
        )

    test_module.Circuit(path)
