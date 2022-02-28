import json

import pytest

import bluepysnap.simulation as test_module
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.frame_report import (
    CompartmentReport,
    PopulationCompartmentReport,
    PopulationSomaReport,
    SomaReport,
)
from bluepysnap.spike_report import PopulationSpikeReport, SpikeReport

from utils import TEST_DATA_DIR


def test_all():
    simulation = test_module.Simulation(str(TEST_DATA_DIR / "simulation_config.json"))
    assert simulation.config["network"] == str(TEST_DATA_DIR / "circuit_config.json")
    assert set(simulation.circuit.nodes) == {"default", "default2"}
    assert set(simulation.circuit.edges) == {"default", "default2"}

    assert simulation.run == {
        "tstop": 1000.0,
        "dt": 0.01,
        "spike_threshold": -15,
        "nsteps_block": 10000,
        "seed": 42,
    }
    assert simulation.time_start == 0.0
    assert simulation.time_stop == 1000.0
    assert simulation.dt == 0.01
    assert simulation.time_units == "ms"

    assert simulation.simulator == "my_simulator"
    assert simulation.conditions == {"celsius": 34.0, "v_init": -80, "other": "something"}

    assert simulation.node_sets.resolved == {"Layer23": {"layer": [2, 3]}}
    assert isinstance(simulation.spikes, SpikeReport)
    assert isinstance(simulation.spikes["default"], PopulationSpikeReport)

    assert set(simulation.reports) == {"soma_report", "section_report"}
    assert isinstance(simulation.reports["soma_report"], SomaReport)
    assert isinstance(simulation.reports["section_report"], CompartmentReport)

    rep = simulation.reports["soma_report"]
    assert set(rep.population_names) == {"default", "default2"}
    assert isinstance(rep["default"], PopulationSomaReport)

    rep = simulation.reports["section_report"]
    assert set(rep.population_names) == {"default", "default2"}
    assert isinstance(rep["default"], PopulationCompartmentReport)


def test_unknown_report():
    simulation = test_module.Simulation(str(TEST_DATA_DIR / "simulation_config.json"))
    simulation.config["reports"] = {
        "soma_report": {"cells": "Layer23", "variable_name": "m", "sections": "unknown"}
    }

    with pytest.raises(BluepySnapError):
        simulation.reports


def test__resolve_config():
    simulation = test_module.Simulation(str(TEST_DATA_DIR / "config.json"))
    assert simulation.config["network"] == str(TEST_DATA_DIR / "circuit_config.json")
    assert set(simulation.circuit.nodes) == {"default", "default2"}

    simulation = test_module.Simulation(str(TEST_DATA_DIR / "config_sim_no_network.json"))
    assert simulation.config["network"] == str(TEST_DATA_DIR / "circuit_config.json")
    assert set(simulation.circuit.nodes) == {"default", "default2"}


def test_no_network_config():
    simulation = test_module.Simulation(str(TEST_DATA_DIR / "simulation_config_no_network.json"))
    with pytest.raises(BluepySnapError):
        simulation.circuit


def test_no_node_set():
    simulation = test_module.Simulation(str(TEST_DATA_DIR / "simulation_config.json"))
    # replace the _config dict with random one that does not contain "node_sets_file" key
    simulation._config = {"key": "value"}
    assert simulation.node_sets == {}


def test__resolve_config_dict():
    input_dict = {"network": "./circuit_config.json", "simulation": "./simulation_config.json"}
    with pytest.raises(BluepySnapError):
        test_module.Simulation(input_dict)

    input_dict = {
        "network": str(TEST_DATA_DIR / "./circuit_config.json"),
        "simulation": str(TEST_DATA_DIR / "./simulation_config.json"),
    }
    simulation = test_module.Simulation(input_dict)
    assert simulation.config["network"] == str(TEST_DATA_DIR / "circuit_config.json")
    assert set(simulation.circuit.nodes) == {"default", "default2"}

    input_dict = json.load(open(str(TEST_DATA_DIR / "./simulation_config.json"), "r"))
    with pytest.raises(BluepySnapError):
        test_module.Simulation(input_dict)

    input_dict["mechanisms_dir"] = "/abspath"
    input_dict["manifest"]["$OUTPUT_DIR"] = str(TEST_DATA_DIR / "reporting")
    input_dict["manifest"]["$INPUT_DIR"] = str(TEST_DATA_DIR)

    simulation = test_module.Simulation(input_dict)
    assert simulation.config["network"] == str(TEST_DATA_DIR / "circuit_config.json")
    assert set(simulation.circuit.nodes) == {"default", "default2"}
