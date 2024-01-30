import contextlib
import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import numpy.testing as npt
import pandas as pd
import pytest

import bluepysnap.simulation_validation as test_module
from bluepysnap.exceptions import BluepySnapValidationError
from bluepysnap.node_sets import NodeSets

from utils import TEST_DATA_DIR, copy_test_data, edit_config


def test__resolve_path():
    config = {"_config_dir": "/fake_root_path"}
    test_path = "/abs/path/config.json"
    assert test_module._resolve_path(test_path, config) == Path(test_path)

    test_path = "relative_config_path.json"
    assert test_module._resolve_path(test_path, config) == Path(f"/fake_root_path/{test_path}")


def test__file_exists():
    assert not test_module._file_exists(None)
    assert not test_module._file_exists(".")
    assert not test_module._file_exists("fake_file")
    assert test_module._file_exists(TEST_DATA_DIR / "circuit_config.json")


def test__get_node_sets():
    circuit_path = str(TEST_DATA_DIR / "circuit_config.json")
    sim_ns_path = str(TEST_DATA_DIR / "node_sets_simple.json")
    circuit_ns_content = json.loads((TEST_DATA_DIR / "node_sets.json").read_text())
    sim_ns_content = json.loads((TEST_DATA_DIR / "node_sets_simple.json").read_text())

    config = {"network": circuit_path, "node_sets_file": sim_ns_path}
    expected_content = {**circuit_ns_content, **sim_ns_content}
    res = test_module._get_node_sets(config)
    assert res.content == expected_content

    assert test_module._get_node_sets({"node_sets_file": sim_ns_path}).content == {**sim_ns_content}
    assert test_module._get_node_sets({"network": circuit_path}).content == {**circuit_ns_content}
    assert test_module._get_node_sets({}).content == {}

    # check that circuit config missing a node sets file works as expected
    with patch.object(test_module, "_parse_config", return_value={}):
        assert test_module._get_node_sets({"network": circuit_path}).content == {}


def test__get_output_dir(tmp_path):
    config = {"_config_dir": tmp_path}
    assert test_module._get_output_dir(config) == tmp_path / "output"

    config = {"_config_dir": tmp_path, "output": {"output_dir": "test_dir"}}
    assert test_module._get_output_dir(config) == tmp_path / "test_dir"

    config = {"_config_dir": tmp_path, "output": {"output_dir": "/tmp/test_dir"}}
    assert test_module._get_output_dir(config) == Path("/tmp/test_dir")


def test__get_circuit_path():
    config = {"_config_dir": "/fake_dir"}
    assert test_module._get_circuit_path(config) == Path("/fake_dir/circuit_config.json")

    config = {"_config_dir": "/fake_dir", "network": ""}
    assert test_module._get_circuit_path(config) == ""

    config = {"network": "fake", "_config_dir": "/fake_dir"}
    assert test_module._get_circuit_path(config) == Path("/fake_dir/fake")

    config = {"network": "/absolute_fake", "_config_dir": "/fake_dir"}
    assert test_module._get_circuit_path(config) == Path("/absolute_fake")


def test__add_validation_parameters():
    config = {}
    test_module._add_validation_parameters(config, "/tmp/config.json")

    assert config["_config_dir"] == Path("/tmp")
    assert config["_output_dir"] == Path("/tmp/output")
    assert config["_node_sets_instance"].content == {}


def test__silent_neurodamus():
    neurodamus = MagicMock()
    setattr(test_module, "neurodamus", neurodamus)

    # Sanity check to check that testing methodology is valid
    stderr = io.StringIO()
    stdout = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        with contextlib.redirect_stdout(stdout):
            print("test_stdout", flush=True)
            print("test_stderr", flush=True, file=sys.stderr)

    assert stdout.getvalue() == "test_stdout\n"
    assert stderr.getvalue() == "test_stderr\n"

    # Check that any output to stdout, stderr is capture
    stderr = io.StringIO()
    stdout = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        with contextlib.redirect_stdout(stdout):
            with test_module._silent_neurodamus() as nd:
                assert nd == neurodamus.core.NeurodamusCore
                print("test_stdout", flush=True)
                print("test_stderr", flush=True, file=sys.stderr)

    assert stdout.getvalue() == ""
    assert stderr.getvalue() == ""


def test__validate_file_exists():
    file_path = TEST_DATA_DIR / "simulation_config.json"
    prefix = "fake"
    assert test_module._validate_file_exists(file_path, prefix) == []

    file_path = TEST_DATA_DIR / "non_existent.file"
    message = f"{prefix}: No such file: {file_path}"
    expected = [BluepySnapValidationError.fatal(message)]
    assert test_module._validate_file_exists(file_path, prefix) == expected

    expected = [BluepySnapValidationError.warning(message)]
    assert test_module._validate_file_exists(file_path, prefix, fatal=False) == expected


def test__validate_node_set_exists():
    node_set = "fake_node_set"
    prefix = "fake"
    config = {"_node_sets_instance": [node_set]}

    assert test_module._validate_node_set_exists(config, node_set, prefix) == []

    node_set = "fail_node_set"
    message = f"{prefix}: Unknown node set: '{node_set}'"
    expected = [BluepySnapValidationError.fatal(message)]
    assert test_module._validate_node_set_exists(config, node_set, prefix) == expected


def test__validate_mechanism_variables():
    @contextlib.contextmanager
    def fake_nd(*_, **__):
        attrs = ["MECH_FOUND", "var_found_MECH_FOUND"]
        yield Mock(h=Mock(__dir__=Mock(return_value=attrs)))

    with patch.object(test_module, "_silent_neurodamus", new=fake_nd):
        name = "TEST_MECH_NOT_FOUND"
        mechanism = {"var_not_found": "fake_value", "var_found": "fake_value"}

        expected_message = f"conditions.mechanisms.{name}: neurodamus: Unknown SUFFIX: {name}"
        expected = [BluepySnapValidationError.fatal(expected_message)]
        assert test_module._validate_mechanism_variables(name, mechanism) == expected

        name = "MECH_FOUND"
        expected_message = (
            f"conditions.mechanisms.{name}: neurodamus: Unknown variable: var_not_found_{name}"
        )
        expected = [BluepySnapValidationError.fatal(expected_message)]
        assert test_module._validate_mechanism_variables(name, mechanism) == expected


def test__validate_mechanisms_no_neurodamus():
    expected_message = (
        "conditions.mechanisms: Can not validate: Neurodamus not found in environment"
    )
    expected = [BluepySnapValidationError.warning(expected_message)]
    assert test_module._validate_mechanisms(mechanisms={}) == expected


@patch.object(test_module, "NEURODAMUS_PRESENT", new=True)
@patch.object(test_module, "_validate_mechanism_variables")
def test__validate_mechanisms(mock_validate_mechanism_variables):
    mock_validate_mechanism_variables.side_effect = lambda x, *_: [x]

    mechanisms = {
        "fake_mech_0": {"fake_field_0": "fake_value_0"},
        "fake_mech_1": {"fake_field_1": "fake_value_1"},
    }
    res = test_module._validate_mechanisms(mechanisms)

    assert res == ["fake_mech_0", "fake_mech_1"]
    mock_validate_mechanism_variables.assert_has_calls(
        [
            call("fake_mech_0", mechanisms["fake_mech_0"]),
            call("fake_mech_1", mechanisms["fake_mech_1"]),
        ]
    )


def test_validate_conditions():
    assert test_module.validate_conditions({}) == []
    config = {
        "_node_sets_instance": NodeSets.from_dict({"fake_node_set": []}),
        "conditions": {
            "modifications": {
                "fake_mod": {"node_set": "fake_node_set"},
                "fail_mod": {"node_set": "fail_node_set"},
            },
            "mechanisms": {"test": {"test_attr": "test_value"}},
        },
    }

    expected = [
        BluepySnapValidationError.fatal(
            "conditions.modifications.fail_mod.node_set: Unknown node set: 'fail_node_set'"
        ),
        BluepySnapValidationError.warning(
            "conditions.mechanisms: Can not validate: Neurodamus not found in environment"
        ),
    ]

    assert test_module.validate_conditions(config) == expected


def test__validate_mod_override_no_neurodamus():
    expected_message = "test: Can not validate: Neurodamus not found in environment"
    expected = [BluepySnapValidationError.warning(expected_message)]
    test_module._validate_mod_override("fake", prefix="test") == expected


@patch.object(test_module, "NEURODAMUS_PRESENT", new=True)
def test__validate_mod_override_with_neurodamus():
    with patch.object(test_module, "_silent_neurodamus"):
        assert test_module._validate_mod_override("fake_override", prefix="fake_prefix") == []

    msg = "neurodamus raised runtime error"

    @contextlib.contextmanager
    def fake_nd(*_, **__):
        yield Mock(load_hoc=Mock(side_effect=RuntimeError(msg)))

    with patch.object(test_module, "_silent_neurodamus", fake_nd):
        res = test_module._validate_mod_override("fake_override", prefix="fake_prefix")

    assert res == [BluepySnapValidationError.fatal(f"fake_prefix: neurodamus: {msg}")]


@patch.object(test_module, "_validate_node_set_exists")
def test__validate_override_unittest(mock_validate_node_set_exists):
    override = {"source": "fake_source", "target": "fake_target"}
    mock_validate_node_set_exists.side_effect = lambda _, node_set, **__: [node_set]

    assert test_module._validate_override(666, override, {}) == ["fake_source", "fake_target"]
    mock_validate_node_set_exists.assert_has_calls(
        [
            call({}, "fake_source", prefix="connection_overrides[666].source"),
            call({}, "fake_target", prefix="connection_overrides[666].target"),
        ]
    )


def test__validate_override():
    override = {"source": "fake_source", "target": "fake_target"}
    config = {"_node_sets_instance": NodeSets.from_dict({"fake_source": [], "fake_target": []})}
    assert test_module._validate_override(0, override, config) == []

    config = {"_node_sets_instance": NodeSets.from_dict({})}
    prefix = "connection_overrides[0]"
    msg = "Unknown node set:"
    expected = [
        BluepySnapValidationError.fatal(f"{prefix}.source: {msg} '{override['source']}'"),
        BluepySnapValidationError.fatal(f"{prefix}.target: {msg} '{override['target']}'"),
    ]

    assert test_module._validate_override(0, override, config) == expected

    override = {"modoverride": "test"}
    expected = [
        BluepySnapValidationError.warning(
            f"{prefix}.modoverride: Can not validate: Neurodamus not found in environment"
        ),
    ]

    assert test_module._validate_override(0, override, {}) == expected


@patch.object(test_module, "_validate_override")
def test_validate_connection_overrides(mock_validate_override):
    assert test_module.validate_connection_overrides({}) == []

    config = {"connection_overrides": list("abc")}
    mock_validate_override.side_effect = lambda i, x, _: (i, x)

    assert test_module.validate_connection_overrides(config) == [0, "a", 1, "b", 2, "c"]

    mock_validate_override.assert_has_calls(
        [call(0, "a", config), call(1, "b", config), call(2, "c", config)]
    )


def test__get_ids_from_spike_file(tmp_path):
    spike_path = tmp_path / "spikes.dat"
    pd.DataFrame({"/scatter": [1]}).to_csv(spike_path, sep="\t")

    assert test_module._get_ids_from_spike_file(spike_path) == {0}

    spike_path = TEST_DATA_DIR / "input_spikes.h5"
    assert test_module._get_ids_from_spike_file(spike_path) == {"default": {0}}

    with pytest.raises(IOError, match=r"Unknown file type: '.fake' \(supported: '.h5', '.dat'\)"):
        test_module._get_ids_from_spike_file("fake_spikes.fake")


def test__get_ids_from_node_set():
    config = {
        "_circuit_config": TEST_DATA_DIR / "circuit_config.json",
        "_node_sets_instance": NodeSets.from_dict({"fake_node_set": {"node_id": [0, 1, 2]}}),
    }

    res = test_module._get_ids_from_node_set("fake_node_set", config)
    expected = {"default": [0, 1, 2], "default2": [0, 1, 2]}
    npt.assert_equal(res, expected)

    # Don't raise if missing properties, expect no populations or ids in result
    config["_node_sets_instance"] = NodeSets.from_dict(
        {"fake_node_set": {"fake_prop": "fake_value"}}
    )
    assert test_module._get_ids_from_node_set("fake_node_set", config) == {}


def test__get_missing_ids():
    nodeset_ids = {"test": [1, 2, 3], "test2": [4, 5]}
    spike_ids_from_dat = {1, 2, 3, 4, 5}
    assert test_module._get_missing_ids(spike_ids_from_dat, nodeset_ids) == []

    spike_ids_from_dat = {1, 3, 5}
    assert test_module._get_missing_ids(spike_ids_from_dat, nodeset_ids) == []

    spike_ids_from_dat.add(6)
    assert test_module._get_missing_ids(spike_ids_from_dat, nodeset_ids) == [6]

    spike_ids_from_h5 = {"test": {1, 2, 3}, "test2": {4, 5}}
    assert test_module._get_missing_ids(spike_ids_from_h5, nodeset_ids) == []

    spike_ids_from_h5 = {"test": {1, 3}, "test2": {5}}
    assert test_module._get_missing_ids(spike_ids_from_h5, nodeset_ids) == []

    spike_ids_from_h5 = {"test": {1, 3, 4}, "test2": {2, 5}}
    expected = [("test", 4), ("test2", 2)]
    assert test_module._get_missing_ids(spike_ids_from_h5, nodeset_ids) == expected

    spike_ids_from_h5 = {"test": {1}, "test2": {5}, "test3": {1, 5}}
    expected = [("test3", 1), ("test3", 5)]
    assert test_module._get_missing_ids(spike_ids_from_h5, nodeset_ids) == expected


@patch.object(test_module, "_get_missing_ids")
def test__compare_ids(mock_missing_ids):
    mock_missing_ids.return_value = []
    assert test_module._compare_ids(None, None, None, None) == []

    source = "fake_superset"
    prefix = "fake_prefix"
    mock_missing_ids.return_value = [0, 1]
    msg = f"{prefix}: 2 id(s) not found in {source}: 0, 1"
    expected = [BluepySnapValidationError.fatal(msg)]
    assert test_module._compare_ids(None, None, source, prefix) == expected

    mock_missing_ids.return_value = [*range(15)]
    msg = f"{prefix}: 15 id(s) not found in {source}: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ..."
    expected = [BluepySnapValidationError.fatal(msg)]
    assert test_module._compare_ids(None, None, source, prefix) == expected


@patch.object(test_module, "_get_ids_from_node_set", new=Mock())
@patch.object(test_module, "_resolve_path", new=Mock())
@patch.object(test_module, "_get_ids_from_spike_file")
@patch.object(test_module, "_get_missing_ids")
def test__validate_spike_file_contents(mock_missing_ids, mock_ids_from_spikes):
    input_config = {"source": "fake_node_set", "spike_file": "fake_spikes.h5"}

    mock_missing_ids.return_value = []
    res = test_module._validate_spike_file_contents(input_config, config=None, prefix="")
    expected = []
    assert res == expected

    mock_missing_ids.return_value = [0, 1, 2]
    res = test_module._validate_spike_file_contents(input_config, config=None, prefix="fake_prefix")
    msg = "fake_prefix: 3 id(s) not found in node set 'fake_node_set': 0, 1, 2"
    expected = [BluepySnapValidationError.fatal(msg)]
    assert res == expected

    mock_missing_ids.return_value = [("fake_population", id_) for id_ in [0, 1, 2]]
    res = test_module._validate_spike_file_contents(input_config, config=None, prefix="fake_prefix")
    msg = (
        "fake_prefix: 3 id(s) not found in node set 'fake_node_set': "
        "('fake_population', 0), ('fake_population', 1), ('fake_population', 2)"
    )
    expected = [BluepySnapValidationError.fatal(msg)]
    assert res == expected

    mock_ids_from_spikes.side_effect = IOError("Unknown", "IOError")
    res = test_module._validate_spike_file_contents(input_config, config=None, prefix="fake_prefix")
    msg = "fake_prefix: Unknown IOError"
    expected = [BluepySnapValidationError.fatal(msg)]
    assert res == expected


def test__validate_spike_input():
    node_sets = NodeSets.from_dict({"fake_node_set": {"node_id": [0]}})

    input_config = {
        "source": "fake_node_set",
        "spike_file": TEST_DATA_DIR / "input_spikes.h5",
    }
    config = {
        "_node_sets_instance": node_sets,
        "_circuit_config": TEST_DATA_DIR / "circuit_config.json",
    }

    assert test_module._validate_spike_input("test", input_config, config) == []

    input_config = {
        "source": "fail_node_set",
        "spike_file": TEST_DATA_DIR / "non_existent.file",
    }

    expected_error_messages = [
        "inputs.test.source: Unknown node set: 'fail_node_set'",
        f"inputs.test.spike_file: No such file: {input_config['spike_file']}",
        "inputs.test.spike_file: Can not validate file contents",
    ]

    expected = [BluepySnapValidationError.fatal(msg) for msg in expected_error_messages]

    assert test_module._validate_spike_input("test", input_config, config) == expected


def test__validate_input_resistance_in_nodes():
    config = {
        "_circuit_config": TEST_DATA_DIR / "circuit_config.json",
        "_node_sets_instance": NodeSets.from_dict({"fake_node_set": {"population": ["default"]}}),
    }
    input_ = {
        "node_set": "fake_node_set",
    }

    res = test_module._validate_input_resistance_in_nodes(input_, config, prefix="fake_prefix")
    assert res == []

    config["_node_sets_instance"] = NodeSets.from_dict({"fake_node_set": {"node_id": [0]}})
    message = "fake_prefix: '@dynamics_params/input_resistance' not found for population 'default2'"
    expected = [BluepySnapValidationError.fatal(message)]
    res = test_module._validate_input_resistance_in_nodes(input_, config, prefix="fake_prefix")
    assert res == expected


@patch.object(test_module, "_validate_node_set_exists")
@patch.object(test_module, "_validate_spike_input")
@patch.object(test_module, "_validate_input_resistance_in_nodes")
def test__validate_input(mock_validate_resistance, mock_validate_spike, mock_validate_node_set):
    mock_validate_spike.return_value = ["fake_spike_error"]
    mock_validate_node_set.return_value = ["fake_nodeset_error"]
    mock_validate_resistance.return_value = ["fake_input_resistance_error"]

    assert test_module._validate_input("test", {}, {}) == []

    input_config = {"node_set": ""}
    assert test_module._validate_input("test", input_config, {}) == ["fake_nodeset_error"]

    input_config = {"module": "synapse_replay"}
    assert test_module._validate_input("test", input_config, {}) == ["fake_spike_error"]

    input_config = {"node_set": "", "module": "synapse_replay"}
    expected = ["fake_nodeset_error", "fake_spike_error"]
    assert test_module._validate_input("test", input_config, {}) == expected

    # check for "can't validate" error if issues with node set validation
    error_msg = "Can not validate presence of '@dynamics_params/input_resistance' in nodes files"
    input_config = {"node_set": "", "module": "shot_noise"}
    expected = [
        "fake_nodeset_error",
        BluepySnapValidationError.fatal(f"inputs.test.shot_noise: {error_msg}"),
    ]
    assert test_module._validate_input("test", input_config, {}) == expected

    # Check that same error is given if no node set is defined
    input_config = {"module": "shot_noise"}
    expected = [BluepySnapValidationError.fatal(f"inputs.test.shot_noise: {error_msg}")]
    assert test_module._validate_input("test", input_config, {}) == expected

    # Check that all relevant modules are checked for input resistance
    modules = [
        "shot_noise",
        "absolute_shot_noise",
        "relative_shot_noise",
        "ornstein_uhlenbeck",
        "relative_ornstein_uhlenbeck ",
    ]
    mock_validate_node_set.return_value = []
    for module in modules:
        input_config = {"node_set": "", "module": module}
        expected = ["fake_input_resistance_error"]
        assert test_module._validate_input("test", input_config, {}) == expected


@patch.object(test_module, "_validate_input")
def test_validate_inputs_unittest(mock_validate_input):
    config = {
        "inputs": {
            "fake_input_0": {"fake_prop_0": "fake_value_0"},
            "fake_input_1": {"fake_prop_1": "fake_value_1"},
        }
    }
    mock_validate_input.side_effect = lambda name, *_: [name]
    assert test_module.validate_inputs(config) == ["fake_input_0", "fake_input_1"]

    mock_validate_input.assert_has_calls(
        [
            call("fake_input_0", {"fake_prop_0": "fake_value_0"}, config),
            call("fake_input_1", {"fake_prop_1": "fake_value_1"}, config),
        ]
    )


def test_validate_inputs():
    node_sets = NodeSets.from_dict({"fake_node_set": {"node_id": [0]}})
    fail_spike_file = TEST_DATA_DIR / "non_existent.file"
    config = {
        "_node_sets_instance": node_sets,
        "_circuit_config": TEST_DATA_DIR / "circuit_config.json",
        "inputs": {
            "pass_0": {"module": "test_module", "node_set": "fake_node_set"},
            "pass_1": {"module": "synapse_replay", "source": "fake_node_set"},
            "pass_2": {
                "module": "synapse_replay",
                "source": "fake_node_set",
                "spike_file": TEST_DATA_DIR / "input_spikes.h5",
            },
            "pass_3": {"module": "not_synapse_replay", "source": "fail_node_set"},
            "pass_4": {"module": "not_synapse_replay", "spike_file": fail_spike_file},
            "fail_0": {"module": "test_module", "node_set": "fail_node_set"},
            "fail_1": {"module": "synapse_replay", "source": "fail_node_set"},
            "fail_2": {"module": "synapse_replay", "spike_file": fail_spike_file},
        },
    }

    expected_error_messages = [
        "inputs.fail_0.node_set: Unknown node set: 'fail_node_set'",
        "inputs.fail_1.source: Unknown node set: 'fail_node_set'",
        f"inputs.fail_2.spike_file: No such file: {fail_spike_file}",
        f"inputs.fail_2.spike_file: Can not validate file contents",
    ]

    expected = [BluepySnapValidationError.fatal(msg) for msg in expected_error_messages]

    assert test_module.validate_inputs(config) == expected


def test_validate_network():
    base_config = {"_config_dir": TEST_DATA_DIR}

    # Test default network
    assert test_module.validate_network(base_config) == []

    path = "circuit_config.json"
    config = {**base_config, "network": path}
    assert test_module.validate_network(config) == []

    path = TEST_DATA_DIR / "circuit_config.json"
    config = {**base_config, "network": path}
    assert test_module.validate_network(config) == []

    path = "fake_path"
    config = {**base_config, "network": path}
    expected = [BluepySnapValidationError.fatal(f"network: No such file: {TEST_DATA_DIR/path}")]
    assert test_module.validate_network(config) == expected

    empty_network = {**base_config, "network": ""}
    expected = [BluepySnapValidationError.warning("network: circuit path not specified")]
    assert test_module.validate_network(empty_network) == expected


@patch.object(test_module, "_validate_node_set_exists")
def test_validate_node_set_unittest(mock_validate_node_set_exists):
    mock_validate_node_set_exists.side_effect = lambda _, node_set, **__: [node_set]
    config = {"node_set": "test"}
    assert test_module.validate_node_set(config) == ["test"]
    mock_validate_node_set_exists.assert_called_once_with(config, "test", prefix="node_set")


def test_validate_node_set():
    config = {"node_set": "test", "_node_sets_instance": NodeSets.from_dict({"test": []})}

    assert test_module.validate_node_set(config) == []
    assert test_module.validate_node_set({}) == []

    config["node_set"] = "fake_node_set"
    expected = [
        BluepySnapValidationError.fatal(f"node_set: Unknown node set: '{config['node_set']}'")
    ]

    assert test_module.validate_node_set(config) == expected


def test_validate_node_sets_file():
    path = TEST_DATA_DIR / "node_sets.json"

    assert test_module.validate_node_sets_file({"node_sets_file": path}) == []

    path = "./fake_path"
    expected = [BluepySnapValidationError.fatal(f"node_sets_file: No such file: {path}")]
    assert test_module.validate_node_sets_file({"node_sets_file": path}) == expected

    assert test_module.validate_node_sets_file({}) == []


def test_validate_output_defaults(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    spikes_file = output_dir / "out.h5"
    spikes_file.touch()

    config = {"_output_dir": output_dir}

    assert test_module.validate_output(config) == []

    spikes_file.unlink()
    expected = [
        BluepySnapValidationError.warning(f"output.spikes_file: No such file: {spikes_file}")
    ]

    assert test_module.validate_output(config) == expected

    output_dir.rmdir()
    expected = [
        BluepySnapValidationError.warning(f"output.output_dir: No such directory: {output_dir}"),
        BluepySnapValidationError.warning(f"output.spikes_file: No such file: {spikes_file}"),
    ]

    assert test_module.validate_output(config) == expected


def test_validate_output_defined_values(tmp_path):
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    spikes_file = output_dir / "test_out.h5"
    spikes_file.touch()
    log_file = output_dir / "test_log.log"
    log_file.touch()

    # check relative path for output_dir
    config = {
        "_output_dir": output_dir,
        "output": {
            "output_dir": output_dir.name,
            "spikes_file": spikes_file.name,
            "log_file": log_file.name,
        },
    }
    assert test_module.validate_output(config) == []

    # Also check with absolute path
    config["output"]["output_dir"] = str(output_dir)
    assert test_module.validate_output(config) == []

    spikes_file.unlink()
    expected = [
        BluepySnapValidationError.warning(f"output.spikes_file: No such file: {spikes_file}")
    ]

    assert test_module.validate_output(config) == expected

    log_file.unlink()
    expected = [
        BluepySnapValidationError.warning(f"output.log_file: No such file: {log_file}"),
        BluepySnapValidationError.warning(f"output.spikes_file: No such file: {spikes_file}"),
    ]

    assert test_module.validate_output(config) == expected

    output_dir.rmdir()
    expected = [
        BluepySnapValidationError.warning(f"output.output_dir: No such directory: {output_dir}"),
        BluepySnapValidationError.warning(f"output.log_file: No such file: {log_file}"),
        BluepySnapValidationError.warning(f"output.spikes_file: No such file: {spikes_file}"),
    ]

    assert test_module.validate_output(config) == expected


def test__validate_report(tmp_path):
    node_sets = NodeSets.from_dict({"fake_node_set": []})
    report_file = tmp_path / "test.h5"
    report_file.touch()

    report = {"cells": "fake_node_set"}
    config = {"_node_sets_instance": node_sets, "_output_dir": tmp_path}

    assert test_module._validate_report("test", report, config) == []

    report = {"file_name": "test"}
    assert test_module._validate_report("test", report, config) == []

    report = {"file_name": "test.h5"}
    assert test_module._validate_report("test", report, config) == []

    report = {"cells": "fail_node_set", "file_name": "fail"}
    fail_path = tmp_path / "fail.h5"
    expected = [
        BluepySnapValidationError.fatal("reports.test.cells: Unknown node set: 'fail_node_set'"),
        BluepySnapValidationError.warning(f"reports.test.file_name: No such file: {fail_path}"),
    ]
    assert test_module._validate_report("test", report, config) == expected


@patch.object(test_module, "_validate_report")
def test_validate_reports_unittest(mock_validate_report):
    config = {
        "reports": {
            "fake_report_0": {"fake_prop_0": "fake_value_0"},
            "fake_report_1": {"fake_prop_1": "fake_value_1"},
        }
    }
    mock_validate_report.side_effect = lambda name, *_: [name]
    assert test_module.validate_reports(config) == ["fake_report_0", "fake_report_1"]

    mock_validate_report.assert_has_calls(
        [
            call("fake_report_0", {"fake_prop_0": "fake_value_0"}, config),
            call("fake_report_1", {"fake_prop_1": "fake_value_1"}, config),
        ]
    )


def test_validate_reports(tmp_path):
    node_sets = NodeSets.from_dict({"fake_node_set": []})
    pass_0_file = tmp_path / "pass_0.h5"
    pass_0_file.touch()
    pass_1_file = tmp_path / "pass_1.h5"
    pass_1_file.touch()
    fail_0_file = tmp_path / "fail_0.h5"
    fail_1_file = tmp_path / "non_existent.h5"
    fail_2_file = tmp_path / "fail_2.h5"

    report = {"cells": "fake_node_set"}
    config = {
        "_node_sets_instance": node_sets,
        "_output_dir": tmp_path,
        "reports": {
            "pass_0": {"cells": "fake_node_set"},
            "pass_1": {"file_name": "pass_1"},
            "pass_2": {"file_name": "pass_1.h5"},
            "fail_0": {"cells": "fake_node_set"},
            "fail_1": {"file_name": "non_existent"},
            "fail_2": {"cells": "fail_node_set"},
        },
    }
    expected = [
        BluepySnapValidationError.warning(f"reports.fail_0.file_name: No such file: {fail_0_file}"),
        BluepySnapValidationError.warning(f"reports.fail_1.file_name: No such file: {fail_1_file}"),
        BluepySnapValidationError.fatal("reports.fail_2.cells: Unknown node set: 'fail_node_set'"),
        BluepySnapValidationError.warning(f"reports.fail_2.file_name: No such file: {fail_2_file}"),
    ]

    assert test_module.validate_reports(config) == expected


def test__get_ids_from_non_virtual_pops():
    config = {"_circuit_config": TEST_DATA_DIR / "circuit_config.json"}
    res = test_module._get_ids_from_non_virtual_pops(config)
    expected = {"default": [0, 1, 2], "default2": [0, 1, 2, 3]}
    npt.assert_equal(res, expected)

    with copy_test_data() as (_, config_path):
        with edit_config(config_path) as circuit_config:
            circuit_config["networks"]["nodes"][0]["populations"]["default2"]["type"] = "virtual"

        config = {"_circuit_config": config_path}
        res = test_module._get_ids_from_non_virtual_pops(config)
        expected = {"default": [0, 1, 2]}
        npt.assert_equal(res, expected)


def test__validate_electrodes_file():
    path = "./fake_path"
    expected = [
        BluepySnapValidationError.fatal(f"run.electrodes_file: No such file: {TEST_DATA_DIR/path}"),
        BluepySnapValidationError.fatal(f"run.electrodes_file: Can not validate file contents"),
    ]
    config = {"run": {"electrodes_file": path}, "_config_dir": TEST_DATA_DIR}
    assert test_module.validate_run(config) == expected

    path = "mock_electrodes_file.h5"
    config = {
        "run": {"electrodes_file": path},
        "_config_dir": TEST_DATA_DIR,
        "_circuit_config": TEST_DATA_DIR / "circuit_config.json",
    }
    assert test_module.validate_run(config) == []

    with patch.object(test_module, "_get_ids_from_non_virtual_pops") as patched:
        patched.return_value = {"default": {0}}
        msg = "run.electrodes_file: 1 id(s) not found in non-virtual populations: ('default', 1)"
        expected = [BluepySnapValidationError.fatal(msg)]
        assert test_module.validate_run(config) == expected

    config["node_set"] = "Node2012"
    config["_node_sets_instance"] = NodeSets.from_file(TEST_DATA_DIR / "node_sets.json")
    assert test_module.validate_run(config) == []

    config["node_set"] = "Layer23"
    msg = "run.electrodes_file: 1 id(s) not found in node set 'Layer23': ('default', 1)"
    expected = [BluepySnapValidationError.fatal(msg)]
    assert test_module.validate_run(config) == expected

    config["_circuit_config"] = ""
    msg = "run.electrodes_file: Can not validate file contents"
    expected = [BluepySnapValidationError.fatal(msg)]
    assert test_module.validate_run(config) == expected


@patch.object(test_module, "_validate_electrodes_file")
def test_validate_run(mock_validate_file):
    mock_validate_file.return_value = []

    assert test_module.validate_run({}) == []
    mock_validate_file.assert_not_called()

    # test with any existing file
    path = TEST_DATA_DIR / "node_sets.json"
    config = {"run": {"electrodes_file": path}}
    assert test_module.validate_run(config) == []
    mock_validate_file.assert_called_once_with(path, config)


@patch.dict(test_module.VALIDATORS)
def test_validate_config_unittest():
    config = {"not_validated_key": "ignored_value"}
    for idx, config_key in enumerate(sorted(test_module.VALIDATORS)):
        test_module.VALIDATORS[config_key] = Mock(return_value=[idx])
        config[config_key] = None

    assert test_module.validate_config(config) == [*range(9)]

    for mock in test_module.VALIDATORS.values():
        mock.assert_called_once_with(config)
        mock.reset_mock()

    # Test that all validators are called even if the key is not in config
    config = {"fake_key": "fake_value"}

    assert test_module.validate_config(config) == [*range(9)]

    for mock in test_module.VALIDATORS.values():
        mock.assert_called_once_with(config)


@patch.object(test_module, "print_validation_errors")
def test_validate(mock_print):
    config_path = TEST_DATA_DIR / "simulation_config.json"

    res = test_module.validate(config_path, print_errors=False)
    mock_print.assert_not_called()
    assert res == set()

    res = test_module.validate(config_path, print_errors=True)
    mock_print.assert_called_once_with([])
    assert res == set()

    with copy_test_data("simulation_config.json") as (data_dir, config_path):
        with edit_config(config_path) as config:
            del config["run"]["random_seed"]

        spikes_file = data_dir / "reporting" / "spikes.h5"
        spikes_file.unlink()

        mock_print.reset_mock()
        res = test_module.validate(str(config_path), print_errors=True)
        expected = [
            BluepySnapValidationError.fatal(
                f"{config_path}:\n\trun: 'random_seed' is a required property"
            ),
            BluepySnapValidationError.warning(f"output.spikes_file: No such file: {spikes_file}"),
        ]
        mock_print.assert_called_once_with(expected)
        assert res == set(expected)
