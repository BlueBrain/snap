import json
import os
import pickle
import warnings

import libsonata
import pytest

import bluepysnap.simulation as test_module
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.frame_report import (
    CompartmentReport,
    PopulationCompartmentReport,
    PopulationSomaReport,
    SomaReport,
)
from bluepysnap.node_sets import NodeSets
from bluepysnap.spike_report import PopulationSpikeReport, SpikeReport

from utils import PICKLED_SIZE_ADJUSTMENT, TEST_DATA_DIR, copy_test_data, edit_config

try:
    Run = libsonata._libsonata.Run
    Conditions = libsonata._libsonata.Conditions
except AttributeError:
    from libsonata._libsonata import SimulationConfig

    Run = SimulationConfig.Run
    Conditions = SimulationConfig.Conditions


def test__warn_on_overwritten_node_sets():
    # No warnings if no overwritten
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        test_module._warn_on_overwritten_node_sets(set())

    with pytest.warns(RuntimeWarning, match="Simulation node sets overwrite 3 .*: a, b, c"):
        test_module._warn_on_overwritten_node_sets(["a", "b", "c"])

    with pytest.warns(RuntimeWarning, match=r"Simulation node sets overwrite 3 .*: a, b, \.\.\."):
        test_module._warn_on_overwritten_node_sets(["a", "b", "c"], print_max=2)


def test_all():
    simulation = test_module.Simulation(str(TEST_DATA_DIR / "simulation_config.json"))
    assert simulation.config["network"] == str(TEST_DATA_DIR / "circuit_config.json")
    assert set(simulation.circuit.nodes) == {"default", "default2"}
    assert set(simulation.circuit.edges) == {"default", "default2"}

    assert isinstance(simulation.run, Run)

    assert simulation.run.tstop == 1000.0
    assert simulation.run.dt == 0.01
    assert simulation.run.spike_threshold == -15
    assert simulation.run.random_seed == 42
    assert simulation.time_start == 0.0
    assert simulation.time_stop == 1000.0
    assert simulation.dt == 0.01
    assert simulation.time_units == "ms"

    assert simulation.simulator == "CORENEURON"
    assert isinstance(simulation.conditions, Conditions)
    assert simulation.conditions.celsius == 34.0
    assert simulation.conditions.v_init == -80

    with pytest.warns(RuntimeWarning, match="Simulation node sets overwrite 1 .* Layer23"):
        assert isinstance(simulation.node_sets, NodeSets)

    expected_content = {
        **json.loads((TEST_DATA_DIR / "node_sets.json").read_text()),
        "Layer23": {"layer": [2, 3]},
        "only_exists_in_simulation": {"node_id": [0, 2]},
    }
    assert simulation.node_sets.content == expected_content

    assert isinstance(simulation.inputs, dict)
    assert isinstance(simulation.spikes, SpikeReport)
    assert isinstance(simulation.spikes["default"], PopulationSpikeReport)

    assert set(simulation.reports) == {"soma_report", "section_report", "lfp_report"}
    assert isinstance(simulation.reports["soma_report"], SomaReport)
    assert isinstance(simulation.reports["section_report"], CompartmentReport)
    assert isinstance(simulation.reports["lfp_report"], CompartmentReport)

    rep = simulation.reports["soma_report"]
    assert set(rep.population_names) == {"default", "default2"}
    assert isinstance(rep["default"], PopulationSomaReport)

    rep = simulation.reports["section_report"]
    assert set(rep.population_names) == {"default", "default2"}
    assert isinstance(rep["default"], PopulationCompartmentReport)


def test_no_warning_when_shared_node_sets_path():
    with copy_test_data(config="simulation_config.json") as (_, config_path):
        circuit = test_module.Simulation(config_path).circuit
        circuit_node_sets_path = circuit.to_libsonata.node_sets_path

        # set simulation node set path = circuit node set path
        with edit_config(config_path) as config:
            config["node_sets_file"] = circuit_node_sets_path

        # Should not raise
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            test_module.Simulation(config_path).node_sets

        # if node_sets_file is not defined, libsonata should use the same path as the circuit
        with edit_config(config_path) as config:
            config.pop("node_sets_file")

        # Should not raise either
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            test_module.Simulation(config_path).node_sets


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


def test_network_file_not_found():
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
            # remove circuit config to prevent libsonata from fetching the path from there
            os.remove(config_path.parent / "circuit_config.json")

        simulation = test_module.Simulation(config_path)
        assert simulation.node_sets.content == {}


def test_pickle(tmp_path):
    pickle_path = tmp_path / "pickle.pkl"
    simulation = test_module.Simulation(TEST_DATA_DIR / "simulation_config.json")

    # trigger some cached properties, to makes sure they aren't being pickeld
    simulation.circuit

    with open(pickle_path, "wb") as fd:
        pickle.dump(simulation, fd)

    with open(pickle_path, "rb") as fd:
        simulation = pickle.load(fd)

    assert pickle_path.stat().st_size < 70 + PICKLED_SIZE_ADJUSTMENT
    assert simulation.dt == 0.01
