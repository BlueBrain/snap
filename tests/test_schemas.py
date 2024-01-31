import re
from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np
import pytest

import bluepysnap.schemas.schemas as test_module
from bluepysnap.exceptions import BluepySnapValidationError

from utils import TEST_DATA_DIR, copy_test_data, edit_config


@pytest.mark.parametrize(
    "type_args",
    (
        ["circuit"],
        ["definitions", "datatypes"],
        ["definitions", "node"],
        ["definitions", "edge"],
        ["node", "biophysical"],
        ["node", "virtual"],
        ["edge", "chemical"],
        ["edge", "electrical"],
        ["edge", "chemical_virtual"],
    ),
)
def test__load_schema_file_exists(type_args):
    assert isinstance(test_module._load_schema_file(*type_args), dict)


def test__load_schema_file_not_existent():
    fake_type = "fake_type"
    with pytest.raises(FileNotFoundError, match=f"Schema file .*{fake_type}.yaml not found"):
        test_module._load_schema_file(fake_type)


def test__validate_schema_for_dict_correct():
    schema = {"properties": {"test": {"required": ["required_field"]}}}
    to_test = {"test": {"required_field": 42}}

    res = test_module._validate_schema_for_dict(schema, to_test)
    assert isinstance(res, Iterable)
    assert len([*res]) == 0


def test__validate_schema_for_dict_incorrect():
    schema = {"properties": {"test": {"required": ["required_field"]}}}
    to_test = {"test": {"these_aren't_the_droids_you're_looking_for": 666}}

    res = test_module._validate_schema_for_dict(schema, to_test)
    assert isinstance(res, Iterable)
    res = [*res]
    assert len(res) == 1
    assert res[0].message == "'required_field' is a required property"


def test__parse_schema_circuit():
    res = test_module._parse_schema("circuit")
    assert "$typedefs" not in res
    assert "$node_file_defs" not in res
    assert "Circuit" in res["title"]


def test__parse_schema_nodes():
    sub_type = "virtual"
    res = test_module._parse_schema("node", sub_type)
    assert "$typedefs" in res
    assert "$node_file_defs" in res
    assert sub_type in res["title"]


def test__parse_schema_edges():
    sub_type = "electrical"
    res = test_module._parse_schema("edge", sub_type)
    assert "$typedefs" in res
    assert "$edge_file_defs" in res
    assert sub_type in res["title"]


def test__parse_schema_unknown():
    obj_type = "fake"
    with pytest.raises(RuntimeError, match=f"Unknown object type: {obj_type}"):
        test_module._parse_schema(obj_type, "")


def test__get_h5_structure_as_dict(tmp_path):
    expected = {
        "group_1": {
            "dataset_1": {"datatype": "int32"},
            "group_2": {
                "dataset_2": {
                    "datatype": "utf-8",
                    "attributes": {"test_attr": "test_value"},
                }
            },
        }
    }
    with h5py.File(tmp_path / "test.h5", "w") as h5:
        h5.create_group("group_1")
        h5["group_1"].create_dataset("dataset_1", data=[0], dtype="i4")
        h5["group_1"].create_group("group_2")
        h5["group_1/group_2"].create_dataset(
            "dataset_2", data="text", dtype=h5py.string_dtype("utf-8")
        )
        h5["group_1/group_2/dataset_2"].attrs.create("test_attr", "test_value")
        res = test_module._get_h5_structure_as_dict(h5)

    assert res == expected


def test__get_h5_structure_as_dict_library_entries(tmp_path):
    # Check that enumerated values types are resolved from @library
    expected = {
        "@library": {
            "dataset_1": {"datatype": "utf-8"},
            "dataset_2": {"datatype": "utf-8"},
        },
        "dataset_1": {"datatype": "utf-8"},
        "dataset_2": {"datatype": "utf-8"},
        "dataset_3": {"datatype": "int64"},
    }
    with h5py.File(tmp_path / "test.h5", "w") as h5:
        lib = h5.create_group("@library")
        h5.create_dataset("dataset_1", data=[0], dtype="i4")
        h5.create_dataset("dataset_2", data=[0], dtype="u4")
        h5.create_dataset("dataset_3", data=[0], dtype="i8")

        lib.create_dataset("dataset_1", data="some text", dtype=h5py.string_dtype("utf-8"))
        lib.create_dataset("dataset_2", data="also, text", dtype=h5py.string_dtype("utf-8"))

        res = test_module._get_h5_structure_as_dict(h5)

    assert res == expected


def test_validate_config_ok():
    config = str(TEST_DATA_DIR / "circuit_config.json")
    res = test_module.validate_circuit_schema("fake_path", config)

    assert len(res) == 0


@pytest.mark.parametrize(
    "to_remove",
    (
        ["components"],
        ["networks", "nodes", 0, "populations", "default", "type"],
        ["networks", "edges", 0, "populations", "default", "type"],
    ),
)
def test_validate_config_ok_missing_optional_fields(to_remove):
    with copy_test_data() as (_, config_copy_path):
        with edit_config(config_copy_path) as config:
            c = config
            for key in to_remove[:-1]:
                c = c[key]
            del c[to_remove[-1]]
        errors = test_module.validate_circuit_schema(str(config_copy_path), config)

        assert len(errors) == 0


@pytest.mark.parametrize(
    ("to_remove_list", "expected"),
    (
        (
            [
                ["networks", "nodes", 0, "populations", "default"],
                ["networks", "nodes", 0, "populations", "default2"],
            ],
            "networks.nodes[0].populations: too few properties",
        ),
        (
            [
                ["networks", "edges", 0, "populations", "default"],
                ["networks", "edges", 0, "populations", "default2"],
            ],
            "networks.edges[0].populations: too few properties",
        ),
        (
            [["networks", "nodes", 0, "populations"]],
            "networks.nodes[0]: 'populations' is a required property",
        ),
        (
            [["networks", "edges", 0, "populations"]],
            "networks.edges[0]: 'populations' is a required property",
        ),
        ([["networks", "nodes", 0]], "networks.nodes: [] should be non-empty"),
        ([["networks", "edges", 0]], "networks.edges: [] should be non-empty"),
        ([["networks", "nodes"]], "networks: 'nodes' is a required property"),
        ([["networks", "edges"]], "networks: 'edges' is a required property"),
        ([["networks"]], "'networks' is a required property"),
    ),
)
def test_validate_config_error(to_remove_list, expected):
    with copy_test_data() as (_, config_copy_path):
        with edit_config(config_copy_path) as config:
            for to_remove in to_remove_list:
                c = config
                for key in to_remove[:-1]:
                    c = c[key]
                del c[to_remove[-1]]
        errors = test_module.validate_circuit_schema(str(config_copy_path), config)

        assert len(errors) == 1
        assert errors[0] == BluepySnapValidationError.fatal(f"{config_copy_path}:\n\t{expected}")


def test_validate_nodes_ok():
    errors = test_module.validate_nodes_schema(
        str(TEST_DATA_DIR / "nodes_single_pop.h5"), "biophysical"
    )
    assert len(errors) == 0


def test_validate_edges_ok():
    errors = test_module.validate_edges_schema(
        str(TEST_DATA_DIR / "edges_single_pop.h5"), "chemical", virtual=False
    )
    assert len(errors) == 0


@pytest.mark.parametrize(
    "missing",
    (
        "nodes",
        "nodes/default/0",
        "nodes/default/0/dynamics_params",
        "nodes/default/0/dynamics_params/holding_current",
        "nodes/default/0/dynamics_params/threshold_current",
        "nodes/default/0/etype",
        "nodes/default/0/model_template",
        "nodes/default/0/model_type",
        "nodes/default/0/morph_class",
        "nodes/default/0/morphology",
        "nodes/default/0/mtype",
        "nodes/default/0/orientation_w",
        "nodes/default/0/orientation_x",
        "nodes/default/0/orientation_y",
        "nodes/default/0/orientation_z",
        "nodes/default/0/synapse_class",
        "nodes/default/0/x",
        "nodes/default/0/y",
        "nodes/default/0/z",
        "nodes/default/node_type_id",
    ),
)
def test_validate_nodes_biophysical_missing_required(missing):
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / "nodes_single_pop.h5"
        with h5py.File(nodes_file, "r+") as h5f:
            del h5f[missing]
        errors = test_module.validate_nodes_schema(str(nodes_file), "biophysical")
        assert len(errors) == 1
        assert f"'{missing.split('/')[-1]}' is a required property" in errors[0].message


@pytest.mark.parametrize(
    "missing",
    (
        "edges",
        "edges/default/0",
        "edges/default/0/afferent_center_x",
        "edges/default/0/afferent_center_y",
        "edges/default/0/afferent_center_z",
        "edges/default/0/afferent_section_id",
        "edges/default/0/afferent_section_pos",
        "edges/default/0/afferent_section_type",
        "edges/default/0/afferent_segment_id",
        "edges/default/0/afferent_segment_offset",
        "edges/default/0/afferent_surface_x",
        "edges/default/0/afferent_surface_y",
        "edges/default/0/afferent_surface_z",
        "edges/default/0/conductance",
        "edges/default/0/decay_time",
        "edges/default/0/delay",
        "edges/default/0/depression_time",
        "edges/default/0/efferent_center_x",
        "edges/default/0/efferent_center_y",
        "edges/default/0/efferent_center_z",
        "edges/default/0/efferent_section_id",
        "edges/default/0/efferent_section_pos",
        "edges/default/0/efferent_section_type",
        "edges/default/0/efferent_segment_id",
        "edges/default/0/efferent_segment_offset",
        "edges/default/0/efferent_surface_x",
        "edges/default/0/efferent_surface_y",
        "edges/default/0/efferent_surface_z",
        "edges/default/0/facilitation_time",
        "edges/default/0/n_rrp_vesicles",
        "edges/default/0/spine_length",
        "edges/default/0/syn_type_id",
        "edges/default/0/u_syn",
        "edges/default/edge_type_id",
        "edges/default/source_node_id",
        "edges/default/target_node_id",
    ),
)
def test_validate_edges_chemical_missing_required(missing):
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / "edges_single_pop.h5"
        with h5py.File(edges_file, "r+") as h5f:
            del h5f[missing]
        errors = test_module.validate_edges_schema(str(edges_file), "chemical", virtual=False)
        assert len(errors) == 1
        assert f"'{missing.split('/')[-1]}' is a required property" in errors[0].message


def test_missing_edge_population():
    with copy_test_data() as (circuit_copy_path, _):
        edges_file = circuit_copy_path / "edges_single_pop.h5"
        with h5py.File(edges_file, "r+") as h5f:
            del h5f["edges/default"]
        errors = test_module.validate_edges_schema(str(edges_file), "chemical", virtual=False)
        assert len(errors) == 1
        assert errors[0] == BluepySnapValidationError.fatal(
            f"{str(edges_file)}:\n\tedges: too few properties"
        )


def test_missing_node_population():
    with copy_test_data() as (circuit_copy_path, _):
        nodes_file = circuit_copy_path / "nodes_single_pop.h5"
        with h5py.File(nodes_file, "r+") as h5f:
            del h5f["nodes/default"]
        errors = test_module.validate_nodes_schema(str(nodes_file), "biophysical")
        assert len(errors) == 1
        assert errors[0] == BluepySnapValidationError.fatal(
            f"{str(nodes_file)}:\n\tnodes: too few properties"
        )


def test_2_edge_populations():
    with copy_test_data() as (circuit_copy_path, _):
        edges_file = circuit_copy_path / "edges_single_pop.h5"
        with h5py.File(edges_file, "r+") as h5f:
            h5f["edges/default2"] = h5f["edges/default"]
        errors = test_module.validate_edges_schema(str(edges_file), "chemical", virtual=False)
        assert len(errors) == 0


def test_2_node_populations():
    with copy_test_data() as (circuit_copy_path, _):
        nodes_file = circuit_copy_path / "nodes_single_pop.h5"
        with h5py.File(nodes_file, "r+") as h5f:
            h5f["nodes/default2"] = h5f["nodes/default"]
        errors = test_module.validate_nodes_schema(str(nodes_file), "biophysical")
        assert len(errors) == 0


def test_virtual_edge_population_ok():
    with copy_test_data() as (circuit_copy_path, _):
        edges_file = circuit_copy_path / "edges_single_pop.h5"
        to_remove = ["efferent_center_x", "efferent_center_y", "efferent_center_z"]
        with h5py.File(edges_file, "r+") as h5f:
            for r in to_remove:
                del h5f[f"edges/default/0/{r}"]
        errors = test_module.validate_edges_schema(str(edges_file), "chemical", virtual=True)
        assert len(errors) == 0


def test_virtual_edge_population_error():
    with copy_test_data() as (circuit_copy_path, _):
        edges_file = circuit_copy_path / "edges_single_pop.h5"
        with h5py.File(edges_file, "r+") as h5f:
            del h5f[f"edges/default/0/afferent_center_x"]
        errors = test_module.validate_edges_schema(str(edges_file), "chemical", virtual=True)
        assert len(errors) == 1
        assert "'afferent_center_x' is a required property" in errors[0].message


def test_virtual_node_population_ok():
    with copy_test_data() as (circuit_copy_path, _):
        nodes_file = circuit_copy_path / "nodes_single_pop.h5"
        to_remove = ["orientation_w", "orientation_x", "orientation_y", "orientation_z"]
        with h5py.File(nodes_file, "r+") as h5f:
            for r in to_remove:
                del h5f[f"nodes/default/0/{r}"]
        errors = test_module.validate_nodes_schema(str(nodes_file), "virtual")
        assert len(errors) == 0


def test_virtual_node_population_error():
    with copy_test_data() as (circuit_copy_path, _):
        nodes_file = circuit_copy_path / "nodes_single_pop.h5"
        with h5py.File(nodes_file, "r+") as h5f:
            del h5f["nodes/default/0/model_type"]
        errors = test_module.validate_nodes_schema(str(nodes_file), "virtual")
        assert len(errors) == 1
        assert "'model_type' is a required property" in errors[0].message


def test_validate_edges_missing_attributes_field():
    # has no attributes at all
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / "edges_single_pop.h5"
        with h5py.File(edges_file, "r+") as h5f:
            del h5f["edges/default/source_node_id"].attrs["node_population"]
        errors = test_module.validate_edges_schema(str(edges_file), "chemical", virtual=False)
        assert len(errors) == 1
        assert "missing required attribute(s) ['node_population']" in errors[0].message


def test_validate_edges_missing_attribute():
    # has attributes but not the required ones
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / "edges_single_pop.h5"
        with h5py.File(edges_file, "r+") as h5f:
            del h5f["edges/default/source_node_id"].attrs["node_population"]
            h5f["edges/default/source_node_id"].attrs.create("population", "some_val")
        errors = test_module.validate_edges_schema(str(edges_file), "chemical", virtual=False)
        assert len(errors) == 1
        assert "missing required attribute(s) ['node_population']" in errors[0].message


@pytest.mark.parametrize(
    "field",
    (
        "edges/default/0/afferent_center_x",
        "edges/default/0/afferent_center_y",
        "edges/default/0/afferent_center_z",
        "edges/default/edge_type_id",
    ),
)
def test_wrong_datatype(field):
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / "edges_single_pop.h5"
        with h5py.File(edges_file, "r+") as h5f:
            del h5f[field]
            h5f.create_dataset(field, data=[0], dtype="i2")
        errors = test_module.validate_edges_schema(str(edges_file), "chemical", virtual=False)
        assert len(errors) == 1
        assert errors[0].level == BluepySnapValidationError.WARNING
        assert f"incorrect datatype 'int16' for '{field}'" in errors[0].message


def test__get_reference_resolver():
    expected = {"const": "test_value"}
    schema = {"$mock_reference": expected}
    resolver = test_module._get_reference_resolver(schema)
    assert resolver.resolve("#/$mock_reference")[1] == expected


def test_nodes_schema_types():
    property_types, dynamics_params = test_module.nodes_schema_types("biophysical")
    assert "x" in property_types
    assert property_types["x"] == np.float32


def test_edges_schema_types():
    edge_property_types = test_module.edges_schema_types("chemical", virtual=True)
    assert "afferent_center_x" in edge_property_types
    assert edge_property_types["afferent_center_x"] == np.float32
