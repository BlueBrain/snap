from pathlib import Path
from unittest.mock import Mock

import h5py
import libsonata
import numpy as np
import pytest

import bluepysnap.neuron_models as test_module
from bluepysnap.circuit import Circuit, CircuitConfig
from bluepysnap.circuit_ids_types import CircuitNodeId
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.sonata_constants import Node

from utils import TEST_DATA_DIR, copy_config, copy_test_data, create_node_population, edit_config


def test_invalid_model_type():
    """test that model type, other than 'biophysical' throws an error"""
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        config = CircuitConfig.from_config(config_copy_path).to_dict()
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
            test_module.NeuronModelsHelper(Mock(), nodes)
        assert "biophysical node population" in e.value.args[0]


def test_get_invalid_node_id():
    nodes = create_node_population(str(TEST_DATA_DIR / "nodes.h5"), "default")
    config = CircuitConfig.from_config(TEST_DATA_DIR / "circuit_config.json").to_dict()
    test_obj = test_module.NeuronModelsHelper(nodes._properties, nodes)

    with pytest.raises(BluepySnapError) as e:
        test_obj.get_filepath("1")
    assert "node_id must be a int or a CircuitNodeId" in e.value.args[0]


def test_get_filepath_biophysical():
    nodes = create_node_population(str(TEST_DATA_DIR / "nodes.h5"), "default")
    config = CircuitConfig.from_config(TEST_DATA_DIR / "circuit_config.json").to_dict()
    test_obj = test_module.NeuronModelsHelper(nodes._properties, nodes)

    node_id = 0
    assert nodes.get(node_id, properties=Node.MODEL_TEMPLATE) == "hoc:small_bio-A"
    actual = test_obj.get_filepath(node_id)
    expected = Path(nodes._properties.biophysical_neuron_models_dir, "small_bio-A.hoc")
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
    expected = Path(nodes._properties.biophysical_neuron_models_dir, "small_bio-C.hoc")
    assert actual == expected


def test_absolute_biophysical_dir():
    with copy_test_data() as (circuit_dir, circuit_config):
        neuron_dir = circuit_dir / "biophysical_neuron_models"
        with h5py.File(str(circuit_dir / "nodes.h5"), "r+") as h5:
            template = [t.decode().split(":") for t in h5["nodes/default/0/model_template"]]
            template = [t[0] + ":" + str(neuron_dir / t[1]) for t in template]
            h5["nodes/default/0/model_template"][...] = template

        nodes = Circuit(circuit_config).nodes["default"]
        test_obj = test_module.NeuronModelsHelper(nodes._properties, nodes)

        for i, t in enumerate(template):
            assert str(test_obj.get_filepath(i)) == ".".join(t.split(":")[::-1])


def test_no_biophysical_dir():
    with copy_test_data() as (data_dir, circuit_config):
        with edit_config(circuit_config) as config:
            del config["components"]["biophysical_neuron_models_dir"]

        with pytest.raises(
            libsonata.SonataError,
            match="Node population .* is defined as 'biophysical' but does not define 'biophysical_neuron_models_dir'",
        ):
            Circuit(circuit_config).nodes["default"]
