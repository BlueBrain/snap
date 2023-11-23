import json
from pathlib import Path
from unittest.mock import Mock, call, patch

import bluepysnap.simulation_validation as test_module
from bluepysnap.exceptions import BluepySnapValidationError
from bluepysnap.node_sets import NodeSets

from utils import TEST_DATA_DIR, copy_test_data, edit_config


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


def test__add_validation_parameters():
    config = {}
    test_module._add_validation_parameters(config, "/tmp/config.json")

    assert config["_config_dir"] == Path("/tmp")
    assert config["_output_dir"] == Path("/tmp/output")
    assert config["_node_sets_instance"].content == {}


def test__validate_file_exists():
    file_path = TEST_DATA_DIR / "simulation_config.json"
    assert test_module._validate_file_exists(file_path) == []

    file_path = TEST_DATA_DIR / "non_existent.file"
    message = f"No such file: {file_path}"
    expected = [BluepySnapValidationError.fatal(message)]
    assert test_module._validate_file_exists(file_path) == expected

    prefix = "test"
    expected = [BluepySnapValidationError.warning(f"{prefix}: {message}")]
    assert test_module._validate_file_exists(file_path, fatal=False, prefix=prefix) == expected


def test__validate_node_set_exists():
    node_set = "fake_node_set"
    config = {"_node_sets_instance": [node_set]}

    assert test_module._validate_node_set_exists(config, node_set) == []

    node_set = "fail_node_set"
    message = f"Unknown node set: '{node_set}'"
    expected = [BluepySnapValidationError.fatal(message)]
    assert test_module._validate_node_set_exists(config, node_set) == expected

    prefix = "test"
    expected = [BluepySnapValidationError.fatal(f"{prefix}: {message}")]
    assert test_module._validate_node_set_exists(config, node_set, prefix=prefix) == expected


def test_validate_conditions():
    assert test_module.validate_conditions({}) == []
    config = {
        "_node_sets_instance": NodeSets.from_dict({"fake_node_set": []}),
        "conditions": {
            "modifications": {
                "fake_mod": {"node_set": "fake_node_set"},
                "fail_mod": {"node_set": "fail_node_set"},
            },
            "mechanisms": "warning",
        },
    }

    expected = [
        BluepySnapValidationError.fatal(
            "conditions.modifications.fail_mod.node_set: Unknown node set: 'fail_node_set'"
        ),
        BluepySnapValidationError.warning(
            "conditions.mechanisms: Validating existence of 'mechanisms' files is not implemented"
        ),
    ]

    assert test_module.validate_conditions(config) == expected


@patch.object(test_module, "_validate_node_set_exists")
def test__validate_override_unittest(mock_validate_node_set_exists):
    override = {"source": "fake_source", "target": "fake_target"}
    mock_validate_node_set_exists.side_effect = lambda _, node_set, **__: [node_set]

    assert test_module._validate_override(666, override, {}) == ["fake_source", "fake_target"]
    mock_validate_node_set_exists.assert_has_calls(
        [
            call({}, "fake_source", prefix="connection_overrides[666]"),
            call({}, "fake_target", prefix="connection_overrides[666]"),
        ]
    )


def test__validate_override():
    override = {"source": "fake_source", "target": "fake_target"}
    config = {"_node_sets_instance": NodeSets.from_dict({"fake_source": [], "fake_target": []})}
    assert test_module._validate_override(0, override, config) == []

    config = {"_node_sets_instance": NodeSets.from_dict({})}
    msg = "connection_overrides[0]: Unknown node set:"
    expected = [
        BluepySnapValidationError.fatal(f"{msg} '{override['source']}'"),
        BluepySnapValidationError.fatal(f"{msg} '{override['target']}'"),
    ]

    assert test_module._validate_override(0, override, config) == expected

    override = {"modoverride": "test"}
    expected = [
        BluepySnapValidationError.warning(
            "connection_overrides[0]: Validating existence of 'modoverride' files is not implemented"
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


def test__validate_input():
    node_sets = NodeSets.from_dict({"fake_node_set": []})
    assert test_module._validate_input("test", {}, {}) == []

    input_config = {
        "node_set": "fake_node_set",
        "module": "synapse_replay",
        "source": "fake_node_set",
        "spike_file": TEST_DATA_DIR / "circuit_config.json",  # any existing path is fine
    }
    config = {"_node_sets_instance": node_sets}

    assert test_module._validate_input("test", input_config, config) == []
    input_config = {
        "node_set": "fail_node_set",
        "module": "synapse_replay",
        "source": "fail_node_set",
        "spike_file": TEST_DATA_DIR / "non_existent.file",
    }

    expected_error_messages = [
        "inputs.test.node_set: Unknown node set: 'fail_node_set'",
        f"inputs.test.spike_file: No such file: {input_config['spike_file']}",
        "inputs.test.source: Unknown node set: 'fail_node_set'",
    ]

    expected = [BluepySnapValidationError.fatal(msg) for msg in expected_error_messages]

    assert test_module._validate_input("test", input_config, config) == expected

    input_config["module"] = "not_synapse_replay"
    expected = [
        BluepySnapValidationError.fatal("inputs.test.node_set: Unknown node set: 'fail_node_set'")
    ]
    assert test_module._validate_input("test", input_config, config) == expected


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
    node_sets = NodeSets.from_dict({"fake_node_set": []})
    mock_spike_file = TEST_DATA_DIR / "circuit_config.json"  # any existing file
    fail_spike_file = TEST_DATA_DIR / "non_existent.file"  # any existing file
    config = {
        "_node_sets_instance": node_sets,
        "inputs": {
            "pass_0": {"module": "test_module", "node_set": "fake_node_set"},
            "pass_1": {"module": "synapse_replay", "source": "fake_node_set"},
            "pass_2": {"module": "synapse_replay", "spike_file": mock_spike_file},
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
    ]

    expected = [BluepySnapValidationError.fatal(msg) for msg in expected_error_messages]

    assert test_module.validate_inputs(config) == expected


def test_validate_network():
    path = TEST_DATA_DIR / "circuit_config.json"

    assert test_module.validate_network({"network": path}) == []

    path = "./fake_path"
    expected = [BluepySnapValidationError.fatal(f"network: No such file: {path}")]
    assert test_module.validate_network({"network": path}) == expected

    expected = [BluepySnapValidationError.warning(f"network: circuit path not specified")]
    assert test_module.validate_network({}) == expected


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
        BluepySnapValidationError.warning(f"output.output_dir: No such directory: {output_dir}")
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
        BluepySnapValidationError.warning(f"output.output_dir: No such directory: {output_dir}")
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


def test_validate_run():
    # test with any existing file
    path = TEST_DATA_DIR / "node_sets.json"

    assert test_module.validate_run({"run": {"electrodes_file": path}}) == []
    assert test_module.validate_run({}) == []

    path = "./fake_path"
    expected = [BluepySnapValidationError.fatal(f"run.electrodes_file: No such file: {path}")]
    assert test_module.validate_run({"run": {"electrodes_file": path}}) == expected


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
