from pathlib import Path

import h5py
import numpy as np
import pytest

import bluepysnap.neuron_models as test_module
from bluepysnap.circuit import Circuit, Config
from bluepysnap.circuit_ids import CircuitNodeId
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.sonata_constants import Node

from utils import TEST_DATA_DIR, copy_config, copy_test_data, create_node_population, edit_config


def test_invalid_model_type():
    """test that model type, other than 'biophysical' or 'point_neuron', throws an error"""
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        config = Config.from_circuit_config(config_copy_path).to_dict()
        nodes_file = config["networks"]["nodes"][0]["nodes_file"]
        with h5py.File(nodes_file, "r+") as h5f:
            grp_name = "nodes/default/0/model_type"
            data = h5f[grp_name][:]
            del h5f[grp_name]
            h5f.create_dataset(
                grp_name, data=["virtual"] * data.shape[0], dtype=h5py.string_dtype()
            )
        nodes = create_node_population(nodes_file, "default")
        with pytest.raises(BluepySnapError) as e:
            test_module.NeuronModelsHelper({}, nodes)
        assert "biophysical or point node population" in e.value.args[0]


def test_get_invalid_node_id():
    nodes = create_node_population(str(TEST_DATA_DIR / "nodes.h5"), "default")
    config = Config.from_circuit_config(TEST_DATA_DIR / "circuit_config.json").to_dict()
    test_obj = test_module.NeuronModelsHelper(config["components"], nodes)

    with pytest.raises(BluepySnapError) as e:
        test_obj.get_filepath("1")
    assert "node_id must be a int or a CircuitNodeId" in e.value.args[0]


def test_get_filepath_biophysical():
    nodes = create_node_population(str(TEST_DATA_DIR / "nodes.h5"), "default")
    config = Config.from_circuit_config(TEST_DATA_DIR / "circuit_config.json").to_dict()
    test_obj = test_module.NeuronModelsHelper(config["components"], nodes)

    node_id = 0
    assert nodes.get(node_id, properties=Node.MODEL_TEMPLATE) == "hoc:small_bio-A"
    actual = test_obj.get_filepath(node_id)
    expected = Path(config["components"]["biophysical_neuron_models_dir"], "small_bio-A.hoc")
    assert actual == expected

    actual = test_obj.get_filepath(np.int64(node_id))
    assert actual == expected
    actual = test_obj.get_filepath(np.uint64(node_id))
    assert actual == expected
    actual = test_obj.get_filepath(np.int32(node_id))
    assert actual == expected
    actual = test_obj.get_filepath(np.uint32(node_id))
    assert actual == expected

    node_id = CircuitNodeId("default", 0)
    assert nodes.get(node_id, properties=Node.MODEL_TEMPLATE) == "hoc:small_bio-A"
    actual = test_obj.get_filepath(node_id)
    assert actual == expected

    node_id = CircuitNodeId("default", 2)
    assert nodes.get(node_id, properties=Node.MODEL_TEMPLATE) == "hoc:small_bio-C"
    actual = test_obj.get_filepath(node_id)
    expected = Path(config["components"]["biophysical_neuron_models_dir"], "small_bio-C.hoc")
    assert actual == expected


def test_no_biophysical_dir():
    nodes = create_node_population(str(TEST_DATA_DIR / "nodes.h5"), "default")
    config = Config.from_circuit_config(TEST_DATA_DIR / "circuit_config.json").to_dict()
    del config["components"]["biophysical_neuron_models_dir"]
    test_obj = test_module.NeuronModelsHelper(config["components"], nodes)

    with pytest.raises(BluepySnapError) as e:
        test_obj.get_filepath(0)
    assert "Missing 'biophysical_neuron_models_dir'" in e.value.args[0]


def test_get_filepath_point():
    with copy_test_data() as (_, config_copy_path):
        with edit_config(config_copy_path) as config:
            config["components"]["point_neuron_models_dir"] = "$COMPONENT_DIR/point_neuron_models"

        circuit = Circuit(config_copy_path)
        nodes = create_node_population(str(TEST_DATA_DIR / "nodes_points.h5"), "default", circuit)
        test_obj = test_module.NeuronModelsHelper(circuit.config["components"], nodes)

        node_id = 0
        assert nodes.get(node_id, properties=Node.MODEL_TEMPLATE) == "nml:empty_bio"
        actual = test_obj.get_filepath(node_id)
        expected = Path(circuit.config["components"]["point_neuron_models_dir"], "empty_bio.nml")
        assert actual == expected

        node_id = 1
        assert nodes.get(node_id, properties=Node.MODEL_TEMPLATE) == "nml:/abs/path/empty_bio"
        actual = test_obj.get_filepath(node_id)
        expected = Path("/abs/path/empty_bio.nml")
        assert actual == expected


def test_no_point_dir():
    nodes = create_node_population(str(TEST_DATA_DIR / "nodes_points.h5"), "default")
    config = Config.from_circuit_config(TEST_DATA_DIR / "circuit_config.json").to_dict()
    test_obj = test_module.NeuronModelsHelper(config["components"], nodes)

    with pytest.raises(BluepySnapError) as e:
        test_obj.get_filepath(0)
    assert "Missing 'point_neuron_models_dir'" in e.value.args[0]
