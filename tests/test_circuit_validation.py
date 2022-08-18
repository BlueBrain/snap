try:
    from unittest.mock import patch, Mock
except ImportError:
    from mock import patch, Mock

import h5py
import numpy as np
import pytest

import bluepysnap.circuit_validation as test_module
from bluepysnap.circuit_validation import Error
from bluepysnap.exceptions import BluepySnapError

from utils import TEST_DATA_DIR, copy_test_data, edit_config


@patch('bluepysnap.schemas.validate_nodes_schema')
@patch('bluepysnap.schemas.validate_edges_schema')
@patch('bluepysnap.schemas.validate_circuit_schema')
def validate(s, *mocks, print_errors=False):
    # reset the mock return value to empty list
    for m in mocks:
        m.return_value = []
    return test_module.validate(s, print_errors=print_errors)


def test_error_comparison():
    err = Error(Error.WARNING, "hello")
    assert err != "hello"


def test_empty_group_size():
    return # TODO: decide on what to do for this
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / "nodes.h5"
        with h5py.File(nodes_file, "r+") as h5f:
            grp = h5f["nodes/default/"].create_group("3")
            with pytest.raises(BluepySnapError):
                test_module._get_group_size(grp)


def test_ok_circuit():
    errors = validate(str(TEST_DATA_DIR / "circuit_config.json"))
    assert errors == set()

    with copy_test_data() as (_, config_copy_path):
        with edit_config(config_copy_path) as config:
            config["networks"]["nodes"][0]["node_types_file"] = None
        errors = validate(str(config_copy_path))
        assert errors == set()

def test_print_errors(capsys):
    with copy_test_data() as (_, config_copy_path):
        with edit_config(config_copy_path) as config:
            del config["networks"]["nodes"]
        validate(str(config_copy_path), print_errors=True)
    assert 'No node population for' in capsys.readouterr().out

@pytest.mark.parametrize(
    'to_remove,expected',
    (
        (['components'], set()),
        (['networks'], set()),
        (
            ['networks', 'nodes'],
            {
                Error(Error.FATAL, 'No node population for "/edges/default/source_node_id"'),
                Error(Error.FATAL, 'No node population for "/edges/default/target_node_id"'),
            }
        ),
        (['networks', 'nodes', 0, 'nodes_file'], set()),
        (
            ['networks', 'nodes', 0, 'populations'],
            {
                Error(Error.FATAL, 'No node population for "/edges/default/source_node_id"'),
                Error(Error.FATAL, 'No node population for "/edges/default/target_node_id"'),
            }
        ),
        (
            ['networks', 'nodes', 0, 'populations', 'default'],
            {
                Error(Error.FATAL, 'No node population for "/edges/default/source_node_id"'),
                Error(Error.FATAL, 'No node population for "/edges/default/target_node_id"'),
            }
        ),
        (['networks', 'edges'], set()),
        (['networks', 'edges', 0, 'edges_file'], set()),
        (['networks', 'edges', 0, 'populations'], set()),
        (['networks', 'edges', 0, 'populations', 'default'], set()),
    ))
def test_missing_data(to_remove, expected):
    with copy_test_data() as (_, config_copy_path):
        with edit_config(config_copy_path) as config:
            c = config
            for key in to_remove[:-1]:
                c = c[key]
            del c[to_remove[-1]]
        errors = validate(str(config_copy_path))
        assert errors == expected


def test_nodes_population_not_found_in_h5():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / "nodes.h5"
        with edit_config(config_copy_path) as config:
            config["networks"]["nodes"][0]["populations"]["fake_population"] = {}
        errors = validate(str(config_copy_path))
        assert errors == {
            Error(
                Error.FATAL,
                f"population 'fake_population' not found in {nodes_file}",
            )
        }


def test_edges_population_not_found_in_h5():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / "edges.h5"
        with edit_config(config_copy_path) as config:
            config["networks"]["edges"][0]["populations"]["fake_population"] = {}
        errors = validate(str(config_copy_path))
        assert errors == {
            Error(
                Error.FATAL,
                f"population 'fake_population' not found in {edges_file}",
            )
        }


def test_ok_node_population_type():
    with copy_test_data() as (_, config_copy_path):
        with edit_config(config_copy_path) as config:
            config["networks"]["nodes"][0]["populations"]["default"]["type"] = "biophysical"
        errors = validate(str(config_copy_path))
        assert errors == set()

@pytest.mark.parametrize('model_type', ('virtual', 'electrical', 'fake_type'))
def test_ok_nonbio_type(model_type):
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        with edit_config(config_copy_path) as config:
            config["networks"]["nodes"][0]["populations"]["default"]["type"] = model_type
        nodes_file = circuit_copy_path / "nodes.h5"
        with h5py.File(nodes_file, "r+") as h5f:
            h5f["nodes/default/0/model_type"][:] = model_type
        errors = validate(str(config_copy_path))
        assert errors == set()



def test_population_type_mismatch():
    with copy_test_data() as (_, config_copy_path):
        fake_type = "fake_type"
        with edit_config(config_copy_path) as config:
            config["networks"]["nodes"][0]["populations"]["default"]["type"] = fake_type
        errors = validate(str(config_copy_path))
        assert errors == {Error(
            Error.WARNING,
            ("Population 'default' type mismatch: "
             f"'biophysical' (nodes_file), '{fake_type}' (config)"))}


def test_ok_edge_population_type():
    with copy_test_data() as (_, config_copy_path):
        with edit_config(config_copy_path) as config:
            config["networks"]["edges"][0]["populations"]["default"]["type"] = "chemical"
        errors = validate(str(config_copy_path))
        assert errors == set()


def test_invalid_edge_population_type():
    with copy_test_data() as (_, config_copy_path):
        fake_type = "fake_type"
        with edit_config(config_copy_path) as config:
            config["networks"]["edges"][0]["populations"]["default"]["type"] = fake_type
        errors = validate(str(config_copy_path))
        assert errors == set()


def test_invalid_config_nodes_file():
    with copy_test_data() as (_, config_copy_path):
        with edit_config(config_copy_path) as config:
            config["networks"]["nodes"][0]["nodes_file"] = "/"
        errors = validate(str(config_copy_path))
        assert errors == {Error(Error.FATAL, 'Invalid "nodes_file": /')}


def test_invalid_config_edges_file():
    with copy_test_data() as (_, config_copy_path):
        with edit_config(config_copy_path) as config:
            config["networks"]["edges"][0]["edges_file"] = "/"
        errors = validate(str(config_copy_path))
        assert errors == {Error(Error.FATAL, 'Invalid "edges_file": /')}


def test_no_nodes_h5():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / "nodes.h5"
        with h5py.File(nodes_file, "r+") as h5f:
            del h5f["nodes"]
        errors = validate(str(config_copy_path))
        assert errors == set()


def test_ok_node_ids_dataset():
    # TODO: check if can be removed
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / "nodes.h5"
        with h5py.File(nodes_file, "r+") as h5f:
            h5f["nodes/default/node_id"] = list(range(len(h5f["nodes/default/node_type_id"])))
        errors = validate(str(config_copy_path))
        assert errors == set()


def test_no_required_node_group_datasets():
    required_datasets = ["model_template", "model_type"]
    for ds in required_datasets:
        with copy_test_data() as (circuit_copy_path, config_copy_path):
            nodes_file = circuit_copy_path / "nodes.h5"
            with h5py.File(nodes_file, "r+") as h5f:
                del h5f["nodes/default/0/" + ds]
            errors = validate(str(config_copy_path))
            assert errors == set()


def test_no_required_bio_node_group_datasets():
    # TODO: check how to combine all these
    required_datasets = sorted(["morphology", "x", "y", "z"])
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / "nodes.h5"
        with h5py.File(nodes_file, "r+") as h5f:
            for ds in required_datasets:
                del h5f["nodes/default/0/" + ds]
        errors = validate(str(config_copy_path))
        assert errors == set()


def test_ok_bio_model_type_in_library():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / "nodes.h5"
        with h5py.File(nodes_file, "r+") as h5f:
            data = h5f["nodes/default/0/model_type"][:]
            del h5f["nodes/default/0/model_type"]
            h5f.create_dataset("nodes/default/0/model_type", data=np.zeros_like(data, dtype=int))
            h5f.create_dataset(
                "nodes/default/0/@library/model_type",
                data=np.array(
                    [
                        "biophysical",
                    ],
                    dtype=h5py.string_dtype(),
                ),
            )
        errors = validate(str(config_copy_path))
        assert errors == set()


def test_no_rotation_bio_node_group_datasets():
    angle_datasets = ["rotation_angle_xaxis", "rotation_angle_yaxis", "rotation_angle_zaxis"]
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / "nodes.h5"
        with h5py.File(nodes_file, "r+") as h5f:
            for ds in angle_datasets:
                del h5f["nodes/default/0/" + ds]
        errors = validate(str(config_copy_path))
        assert errors == set()


def test_no_bio_component_dirs():
    dirs = ["morphologies_dir", "biophysical_neuron_models_dir"]
    for dir_ in dirs:
        with copy_test_data() as (_, config_copy_path):
            with edit_config(config_copy_path) as config:
                del config["components"][dir_]
            errors = validate(str(config_copy_path))
            # multiplication by 2 because we have 2 populations, each produces the same error.
            assert errors == set()


def test_invalid_bio_alternate_morphology_dir():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        component = "neurolucida-asc"
        fake_path = str(circuit_copy_path / "fake/path")
        with edit_config(config_copy_path) as config:
            config["networks"]["nodes"][0]["populations"]["default"]["alternate_morphologies"] = {
                component: fake_path
            }
        errors = validate(str(config_copy_path))
        assert errors == {
            Error(Error.FATAL, f'Invalid components "{component}": {fake_path}')
        }


@patch("bluepysnap.circuit_validation.MAX_MISSING_FILES_DISPLAY", 1)
def test_no_morph_files():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / "nodes.h5"
        with h5py.File(nodes_file, "r+") as h5f:
            h5f["nodes/default/0/morphology"][0] = "noname"
        errors = validate(str(config_copy_path))
        assert errors == {
            Error(
                Error.WARNING,
                f"missing 1 files in group morphology: default/0[{nodes_file}]:\n\tnoname.swc\n",
            )
        }

        with h5py.File(nodes_file, "r+") as h5f:
            morph = h5f["nodes/default/0/morphology"]
            morph[:] = ["noname" + str(i) for i in range(len(morph))]
        errors = validate(str(config_copy_path))
        assert errors == {
            Error(
                Error.WARNING,
                f"missing 3 files in group morphology: default/0[{nodes_file}]:"
                "\n\tnoname0.swc\n\t...\n",
            )
        }


def test_no_alternate_morph_files():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        nodes_file = str(circuit_copy_path / "nodes.h5")
        with edit_config(config_copy_path) as config:
            config["networks"]["nodes"][0]["populations"]["default"]["alternate_morphologies"] = {
                "neurolucida-asc": config["components"]["morphologies_dir"]
            }
        errors = validate(str(config_copy_path))
        assert errors == {
            Error(
                Error.WARNING,
                f"missing 1 files in group morphology: default/0[{nodes_file}]:\n\tmorph-A.asc\n",
            )
        }


@patch("bluepysnap.circuit_validation.MAX_MISSING_FILES_DISPLAY", 1)
def test_no_morph_library_files():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / "nodes.h5"
        with h5py.File(nodes_file, "r+") as h5f:
            grp = h5f["nodes/default/0"]
            str_dtype = h5py.special_dtype(vlen=str)
            grp.create_dataset("@library/morphology", shape=(1,), dtype=str_dtype)
            grp["@library/morphology"][:] = "noname"
            shape = grp["morphology"].shape
            del grp["morphology"]
            grp.create_dataset("morphology", shape=shape, fillvalue=0)
        errors = validate(str(config_copy_path))
        assert errors == {
            Error(
                Error.WARNING,
                f"missing 1 files in group morphology: default/0[{nodes_file}]:\n\tnoname.swc\n",
            )
        }


def test_no_template_files():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / "nodes.h5"
        with h5py.File(nodes_file, "r+") as h5f:
            h5f["nodes/default/0/model_template"][0] = "hoc:noname"
        errors = validate(str(config_copy_path))
        assert errors == {
            Error(
                Error.WARNING,
                f"missing 1 files in group model_template: default/0[{nodes_file}]:\n\tnoname.hoc\n",
            )
        }


@patch("bluepysnap.circuit_validation.MAX_MISSING_FILES_DISPLAY", 1)
def test_no_template_library_files():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / "nodes.h5"
        with h5py.File(nodes_file, "r+") as h5f:
            grp = h5f["nodes/default/0"]
            str_dtype = h5py.special_dtype(vlen=str)
            grp.create_dataset("@library/model_template", shape=(1,), dtype=str_dtype)
            grp["@library/model_template"][:] = "hoc:noname"
            shape = grp["model_template"].shape
            del grp["model_template"]
            grp.create_dataset("model_template", shape=shape, fillvalue=0)
        errors = validate(str(config_copy_path))
        assert errors == {
            Error(
                Error.WARNING,
                f"missing 1 files in group model_template: default/0[{nodes_file}]:\n\tnoname.hoc\n",
            )
        }


def test_no_edges_h5():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / "edges.h5"
        with h5py.File(edges_file, "r+") as h5f:
            del h5f["edges"]
        errors = validate(str(config_copy_path))
        assert errors == set()


def test_no_edge_group():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / "edges.h5"
        with h5py.File(edges_file, "r+") as h5f:
            del h5f["edges/default/0"]
        errors = validate(str(config_copy_path))
        assert errors == set()


def test_no_edge_group_missing_requiered_datasets():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        required_datasets = sorted(["edge_type_id", "source_node_id", "target_node_id"])
        edges_file = circuit_copy_path / "edges.h5"
        with h5py.File(edges_file, "r+") as h5f:
            del h5f["edges/default/0"]
            for ds in required_datasets:
                del h5f["edges/default/" + ds]
        errors = validate(str(config_copy_path))
        assert errors == set()


def test_no_edge_group_no_optional_datasets():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        optional_datasets = sorted(["edge_group_id", "edge_group_index"])
        edges_file = circuit_copy_path / "edges.h5"
        with h5py.File(edges_file, "r+") as h5f:
            del h5f["edges/default/0"]
            for ds in optional_datasets:
                del h5f["edges/default/" + ds]
        errors = validate(str(config_copy_path))
        assert errors == set()


def test_no_required_edge_population_datasets_one_group():
    required_datasets = sorted(["edge_type_id", "source_node_id", "target_node_id"])
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / "edges.h5"
        with h5py.File(edges_file, "r+") as h5f:
            for ds in required_datasets:
                del h5f["edges/default/" + ds]
        errors = validate(str(config_copy_path))
        assert errors == set()


def test_missing_optional_edge_population_datasets_one_group():
    optional_datasets = sorted(["edge_group_id", "edge_group_index"])
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / "edges.h5"
        with h5py.File(edges_file, "r+") as h5f:
            for ds in optional_datasets:
                del h5f["edges/default/" + ds]
        errors = validate(str(config_copy_path))
        assert errors == set()


def test_no_required_edge_population_datasets_multiple_groups():
    required_datasets = sorted(
        ["edge_type_id", "source_node_id", "target_node_id", "edge_group_id", "edge_group_index"]
    )
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / "edges.h5"
        with h5py.File(edges_file, "r+") as h5f:
            for ds in required_datasets:
                del h5f["edges/default/" + ds]
            h5f.create_group("edges/default/1")
        errors = validate(str(config_copy_path))
        assert errors == set()


def test_edge_population_multiple_groups():
    # TODO: decide what to do with this
    return
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / "edges.h5"
        with h5py.File(edges_file, "r+") as h5f:
            h5f.create_group("edges/default/1")
        errors = validate(str(config_copy_path))
        assert (
            Error(
                Error.WARNING,
                f"Population default of {edges_file} have multiple groups. "
                "Cannot be read via bluepysnap or libsonata",
            )
            in errors
        )


def test_edge_population_missing_edge_group_id_one_group():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / "edges.h5"
        with h5py.File(edges_file, "r+") as h5f:
            del h5f["edges/default/edge_group_id"]
        errors = validate(str(config_copy_path))
        assert errors == set()


def test_edge_population_missing_edge_group_index_one_group():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / "edges.h5"
        with h5py.File(edges_file, "r+") as h5f:
            del h5f["edges/default/edge_group_index"]
        errors = validate(str(config_copy_path))
        assert errors == set()


def test_edge_population_missing_edge_group_id_index_one_group():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / "edges.h5"
        with h5py.File(edges_file, "r+") as h5f:
            del h5f["edges/default/edge_group_index"]
            del h5f["edges/default/edge_group_id"]
        errors = validate(str(config_copy_path))
        assert errors == set()


def test_no_required_bbp_edge_group_datasets():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / "edges.h5"
        with h5py.File(edges_file, "r+") as h5f:
            del h5f["edges/default/0/syn_weight"]
        errors = validate(str(config_copy_path))
        assert errors == set()


def test_no_edge_indices():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / "edges.h5"
        with h5py.File(edges_file, "r+") as h5f:
            del h5f["edges/default/indices"]
        errors = validate(str(config_copy_path))
        assert errors == {
            Error(Error.WARNING, f'No "indices" in {edges_file}'),
        }


def test_no_edge_source_to_target():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / "edges.h5"
        with h5py.File(edges_file, "r+") as h5f:
            del h5f["edges/default/indices/source_to_target"]
            del h5f["edges/default/indices/target_to_source"]
        errors = validate(str(config_copy_path))
        assert errors == {
            Error(Error.WARNING, f'No "source_to_target" in {edges_file}'),
            Error(Error.WARNING, f'No "target_to_source" in {edges_file}')
        }


def test_no_edge_all_node_ids():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        nodes_file = circuit_copy_path / "nodes.h5"
        with h5py.File(nodes_file, "r+") as h5f:
            del h5f["nodes/default/0"]
        errors = validate(str(config_copy_path))
        assert errors == {
            Error(
                Error.FATAL,
                "/edges/default/source_node_id does not have node ids in its node population",
            ),
            Error(
                Error.FATAL,
                "/edges/default/target_node_id does not have node ids in its node population",
            ),
        }


def test_invalid_edge_node_ids():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / "edges.h5"
        with h5py.File(edges_file, "r+") as h5f:
            h5f["edges/default/source_node_id"][0] = 99999
            h5f["edges/default/target_node_id"][0] = 99999
        errors = validate(str(config_copy_path))
        assert errors == {
            Error(
                Error.FATAL,
                "/edges/default/source_node_id misses node ids in its node population: [99999]",
            ),
            Error(
                Error.FATAL,
                "/edges/default/target_node_id misses node ids in its node population: [99999]",
            ),
            Error(
                Error.FATAL,
                f"Population {edges_file} edges [99999] have node ids [0 1] instead of "
                "single id 2",
            ),
            Error(
                Error.FATAL,
                f"Population {edges_file} edges [99999] have node ids [0 1] instead of "
                "single id 0",
            ),
        }


def test_explicit_edges_no_node_population_attr():
    with copy_test_data() as (circuit_copy_path, config_copy_path):
        edges_file = circuit_copy_path / "edges.h5"
        with h5py.File(edges_file, "r+") as h5f:
            del h5f["edges/default/source_node_id"].attrs["node_population"]
        errors = validate(str(config_copy_path))
        assert errors == set()


def test_no_duplicate_population_names():
    with copy_test_data() as (_, config_copy_path):
        with edit_config(config_copy_path) as config:
            config["networks"]["nodes"].append(config["networks"]["nodes"][0])
        errors = validate(str(config_copy_path))
        assert errors == {
            Error(Error.FATAL, 'Already have population "default" in config for type "nodes"')
        }
