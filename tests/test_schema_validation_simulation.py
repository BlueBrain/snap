from copy import deepcopy
from unittest.mock import Mock, patch

import pytest

import bluepysnap.schemas.schemas as test_module

CONFIG_FILE = "fake_simulation.json"
SCHEMA = test_module._parse_schema("simulation")

MINIMUM_MANDATORY = {"run": {"tstop": 0.1, "dt": 0.1, "random_seed": 1}}

MANDATORY_FOR_OPTIONALS = {
    **MINIMUM_MANDATORY,
    "conditions": {
        "modifications": {
            "TTX": {"node_set": "fake", "type": "TTX"},
            "ConfigureAllSections": {
                "node_set": "fake",
                "type": "ConfigureAllSections",
                "section_configure": "fake",
            },
        }
    },
    "reports": {
        "fake": {
            "type": "compartment",
            "variable_name": "fake",
            "dt": 0.1,
            "start_time": 0.1,
            "end_time": 0.1,
        }
    },
    "connection_overrides": [
        {
            "name": "fake",
            "source": "fake",
            "target": "fake",
        }
    ],
}


def _get_fake_inputs(data):
    """Creates an `inputs` entry with provided data and a few required properties"""
    return {
        "inputs": {
            "fake_input": {"delay": 0.1, "duration": 0.1, "node_set": "fake_node_set", **data}
        }
    }


def _get_remove_paths(config):
    """Helper function to get paths to the leaf items in a nested dictionary."""
    paths = []

    def _recurse_paths(config, path=None):
        path = path or []
        for key, value in config.items():
            if isinstance(value, dict):
                _recurse_paths(config[key], path + [key])
            elif isinstance(value, list):
                _recurse_paths(config[key][0], path + [key, 0])
            else:
                paths.append(path + [key])

    _recurse_paths(config)
    return paths


@patch.object(test_module, "_parse_schema", new=Mock(return_value=SCHEMA))
def _validate(config):
    """
    Helper function to validate the schema.

    Schema mocked to not have to parse it from file for every validation.
    """
    return test_module.validate_circuit_schema(CONFIG_FILE, config)


def _remove_from_config(config, to_remove):
    """Helper function to remove a value from a dictionary."""
    for key in to_remove[:-1]:
        config = config[key]
    del config[to_remove[-1]]


@pytest.mark.parametrize("config", [MINIMUM_MANDATORY, MANDATORY_FOR_OPTIONALS])
def test_all_mandatory_fields_pass(config):
    """Check that the defined mandatories pass"""
    assert _validate(config) == []


@pytest.mark.parametrize("to_remove", [["run"], *_get_remove_paths(MANDATORY_FOR_OPTIONALS)])
def test_all_mandatory_fields_missing(to_remove):
    """
    Test that error is reported for all mandatory fields.

    Inputs' mandatory fields are tested separately.
    """
    config = deepcopy(MANDATORY_FOR_OPTIONALS)
    _remove_from_config(config, to_remove)

    expected = f"{to_remove[-1]}' is a required property"
    res = _validate(config)
    assert len(res) == 1
    assert expected in res[0].message


@pytest.mark.parametrize(
    "input_data",
    [
        {"module": "linear", "input_type": "current_clamp", "amp_start": 0.1},
        {"module": "relative_linear", "input_type": "current_clamp", "percent_start": 0.1},
        {
            "module": "pulse",
            "input_type": "current_clamp",
            "amp_start": 0.1,
            "width": 0.1,
            "frequency": 0.1,
        },
        {"module": "subthreshold", "input_type": "current_clamp", "percent_less": 1},
        {"module": "synapse_replay", "input_type": "spikes", "spike_file": "fake"},
        {"module": "seclamp", "input_type": "voltage_clamp", "voltage": 0.1},
        {"module": "noise", "input_type": "current_clamp", "mean": 0.1},
        {"module": "noise", "input_type": "current_clamp", "mean_percent": 0.1},
        {
            "module": "shot_noise",
            "input_type": "current_clamp",
            "rise_time": 0.1,
            "decay_time": 0.1,
            "rate": 0.1,
            "amp_mean": 0.1,
            "amp_var": 0.1,
        },
        {
            "module": "shot_noise",
            "input_type": "conductance",
            "rise_time": 0.1,
            "decay_time": 0.1,
            "rate": 0.1,
            "amp_mean": 0.1,
            "amp_var": 0.1,
        },
        {
            "module": "absolute_shot_noise",
            "input_type": "current_clamp",
            "rise_time": 0.1,
            "decay_time": 0.1,
            "rate": 0.1,
            "amp_cv": 0.1,
            "mean": 0.1,
            "sigma": 0.1,
        },
        {
            "module": "absolute_shot_noise",
            "input_type": "conductance",
            "rise_time": 0.1,
            "decay_time": 0.1,
            "rate": 0.1,
            "amp_cv": 0.1,
            "mean": 0.1,
            "sigma": 0.1,
        },
        {
            "module": "relative_shot_noise",
            "input_type": "current_clamp",
            "rise_time": 0.1,
            "decay_time": 0.1,
            "rate": 0.1,
            "amp_cv": 0.1,
            "mean_percent": 0.1,
            "sd_percent": 0.1,
        },
        {
            "module": "relative_shot_noise",
            "input_type": "conductance",
            "rise_time": 0.1,
            "decay_time": 0.1,
            "rate": 0.1,
            "amp_cv": 0.1,
            "mean_percent": 0.1,
            "sd_percent": 0.1,
        },
        {
            "module": "ornstein_uhlenbeck",
            "input_type": "current_clamp",
            "tau": 0.1,
            "mean": 0.1,
            "sigma": 0.1,
        },
        {
            "module": "ornstein_uhlenbeck",
            "input_type": "conductance",
            "tau": 0.1,
            "mean": 0.1,
            "sigma": 0.1,
        },
        {
            "module": "relative_ornstein_uhlenbeck",
            "input_type": "current_clamp",
            "tau": 0.1,
            "mean_percent": 0.1,
            "sd_percent": 0.1,
        },
        {
            "module": "relative_ornstein_uhlenbeck",
            "input_type": "conductance",
            "tau": 0.1,
            "mean_percent": 0.1,
            "sd_percent": 0.1,
        },
    ],
)
def test_inputs_expected_values(input_data):
    """Test that error is reported on missing mandatory fields and on wrong `input_type`."""
    tested = _get_fake_inputs(input_data)
    module, input_type = input_data["module"], input_data["input_type"]

    config = {**MINIMUM_MANDATORY, **tested}
    assert _validate(config) == []

    for entry in _get_remove_paths(tested):
        conf = deepcopy(config)
        _remove_from_config(conf, entry)
        res = _validate(conf)
        assert len(res) == 1

        removed = entry[-1]

        # in the case of `noise` either `mean` or `mean_percent` is required. If neither is given
        # message reports that the input definition is not valid under any of the schemas.
        if module == "noise" and removed in ("mean", "mean_percent"):
            assert "either 'mean' or 'mean_percent' is required (not both)" in res[0].message
        else:
            assert f"'{removed}' is a required property" in res[0].message

    wrong_type = "extracellular_stimulation"
    conf = {**MINIMUM_MANDATORY, **_get_fake_inputs({**input_data, "input_type": wrong_type})}
    res = _validate(conf)
    assert len(res) == 1

    # if there's multiple possible input types for a module the message reports that
    # [...].input_type: 'fake_type' is not valid under any of the given schemas
    if "shot_noise" in module or "ornstein_uhlenbeck" in module:
        assert (
            f"input_type: '{wrong_type}' is not one of ['current_clamp', 'conductance']"
            in res[0].message
        )
    else:
        assert f"input_type: '{input_type}' was expected" in res[0].message


def test_run():
    config = deepcopy(MINIMUM_MANDATORY)
    config["run"].update(
        {
            "spike_threshold": 1,
            "integration_method": "0",
            "stimulus_seed": 1,
            "ionchannel_seed": 1,
            "minis_seed": 1,
            "synapse_seed": 1,
            "electrodes_file": "fake_file",
        }
    )
    assert _validate(config) == []

    config["run"]["integration_method"] = "fail"

    res = _validate(config)
    assert "integration_method: 'fail' is not one of ['0', '1', '2']" in res[0].message


def test_output():
    config = {
        **MINIMUM_MANDATORY,
        "output": {
            "output_dir": "fake",
            "log_file": "fake",
            "spikes_file": "fake",
            "spikes_sort_order": "none",
        },
    }
    assert _validate(config) == []

    config["output"]["spikes_sort_order"] = "fail"
    res = _validate(config)
    assert len(res) == 1
    assert "spikes_sort_order: 'fail' is not one of ['by_id', 'by_time', 'none']" in res[0].message


def test_conditions():
    config = {
        **MINIMUM_MANDATORY,
        "conditions": {
            "celsius": 0.1,
            "v_init": 0.1,
            "spike_location": "soma",
            "extracellular_calcium": 0.1,
            "randomize_gaba_rise_time": True,
            "mechanisms": {"fake_mechanism": {"fake_property": "fake"}},
            "modifications": {
                "fake_modification": {
                    "node_set": "fake_node_set",
                    "type": "TTX",
                }
            },
        },
    }
    assert _validate(config) == []

    config["conditions"]["spike_location"] = "fail_0"
    config["conditions"]["modifications"]["fake_modification"]["type"] = "fail_1"

    res = _validate(config)
    assert len(res) == 1
    assert "spike_location: 'fail_0' is not one of ['AIS', 'soma']" in res[0].message
    assert "type: 'fail_1' is not one of ['ConfigureAllSections', 'TTX']" in res[0].message


def test_reports():
    config = deepcopy(MANDATORY_FOR_OPTIONALS)
    config["reports"]["fake"].update(
        {
            "cells": "fake_node_set",
            "sections": "soma",
            "scaling": "none",
            "compartments": "center",
            "unit": "fake_unit",
            "file_name": "fake_file",
            "enabled": True,
        }
    )
    assert _validate(config) == []

    config["reports"]["fake"]["sections"] = "fail_0"
    config["reports"]["fake"]["scaling"] = "fail_1"
    config["reports"]["fake"]["compartments"] = "fail_2"

    res = _validate(config)
    assert len(res) == 1
    assert (
        "sections: 'fail_0' is not one of ['all', 'apic', 'axon', 'dend', 'soma']" in res[0].message
    )
    assert "scaling: 'fail_1' is not one of ['area', 'none']" in res[0].message
    assert "compartments: 'fail_2' is not one of ['all', 'center']" in res[0].message


def test_inputs():
    input_properties = {
        "module": "noise",
        "input_type": "current_clamp",
        "mean": 0.1,
        "amp_cv": 0.1,
        "amp_end": 0.1,
        "amp_mean": 0.1,
        "amp_start": 0.1,
        "amp_var": 0.1,
        "decay_time": 0.1,
        "dt": 0.1,
        "frequency": 0.1,
        "percent_end": 0.1,
        "percent_less": 1,
        "percent_start": 0.1,
        "random_seed": 1,
        "rate": 0.1,
        "reversal": 0.1,
        "rise_time": 0.1,
        "sd_percent": 0.1,
        "series_resistance": 0.1,
        "sigma": 0.1,
        "spike_file": "fake_file",
        "source": "fake_node_set",
        "tau": 0.1,
        "variance": 0.1,
        "voltage": 0.1,
        "width": 0.1,
    }
    input_conf = _get_fake_inputs(input_properties)
    config = {**MINIMUM_MANDATORY, **input_conf}

    assert _validate(config) == []

    # `mean` and `mean_percent` can not both be defined
    config["inputs"]["fake_input"]["mean_percent"] = 0.1
    res = _validate(config)
    assert len(res) == 1
    assert "fake_input: either 'mean' or 'mean_percent' is required (not both)" in res[0].message
