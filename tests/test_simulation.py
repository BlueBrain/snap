import json
import os

import libsonata
import pytest
from libsonata import SonataError

import bluepysnap.simulation as test_module
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.frame_report import (
    CompartmentReport,
    PopulationCompartmentReport,
    PopulationSomaReport,
    SomaReport,
)
from bluepysnap.spike_report import PopulationSpikeReport, SpikeReport

from utils import TEST_DATA_DIR, copy_test_data, edit_config


def test_all():
    simulation = test_module.Simulation(str(TEST_DATA_DIR / "simulation_config.json"))
    assert simulation.config["network"] == str(TEST_DATA_DIR / "circuit_config.json")
    assert set(simulation.circuit.nodes) == {"default", "default2"}
    assert set(simulation.circuit.edges) == {"default", "default2"}

    assert isinstance(simulation.run, libsonata._libsonata.Run)
    assert simulation.run.tstop == 1000.0
    assert simulation.run.dt == 0.01
    assert simulation.run.spike_threshold == -15
    assert simulation.run.random_seed == 42
    assert simulation.time_start == 0.0
    assert simulation.time_stop == 1000.0
    assert simulation.dt == 0.01
    assert simulation.time_units == "ms"

    assert simulation.simulator == "CORENEURON"
    assert isinstance(simulation.conditions, libsonata._libsonata.Conditions)
    assert simulation.conditions.celsius == 34.0
    assert simulation.conditions.v_init == -80

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
    with copy_test_data(config="simulation_config.json") as (_, config_path):
        with edit_config(config_path) as config:
            config["reports"]["soma_report"]["sections"] = "unknown"

        with pytest.raises(libsonata.SonataError, match="Invalid value.*for key 'sections'"):
            test_module.Simulation(config_path)


def test_nonimplemented_report():
    with copy_test_data(config="simulation_config.json") as (_, config_path):
        with edit_config(config_path) as config:
            config["reports"]["soma_report"]["sections"] = "axon"

        with pytest.raises(
            BluepySnapError, match="Report soma_report: format axon not yet supported"
        ):
            test_module.Simulation(config_path).reports


def test_no_network_config():
    with copy_test_data(config="simulation_config.json") as (_, config_path):
        with edit_config(config_path) as config:
            config.pop("network")

        os.remove(config_path.parent / "circuit_config.json")
        simulation = test_module.Simulation(config_path)

        with pytest.raises(BluepySnapError, match="'network' file not found"):
            simulation.circuit


def test_no_node_set():
    with copy_test_data(config="simulation_config.json") as (_, config_path):
        with edit_config(config_path) as config:
            config.pop("node_sets_file")
            os.remove(config_path.parent / "circuit_config.json")

        simulation = test_module.Simulation(config_path)
        assert simulation.node_sets == {}
