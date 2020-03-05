import pytest

from bluepysnap.exceptions import BluepySnapError
import bluepysnap.simulation as test_module
from bluepysnap.spike_report import SpikeReport, PopulationSpikeReport
from bluepysnap.frame_report import (SomaReport, PopulationSomaReport,
                                     SectionReport, PopulationSectionReport)


from utils import TEST_DATA_DIR


def test_all():
    simulation = test_module.Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
    assert simulation.config["network"] == str(TEST_DATA_DIR / 'circuit_config.json')
    assert sorted(list(simulation.circuit.nodes)) == ['default', 'default2']
    assert sorted(list(simulation.circuit.nodes)) == ['default', 'default2']
    assert list(simulation.circuit.edges) == ['default']

    assert simulation.run == {"tstop": 1000.0, "dt": 0.01, "spike_threshold": -15,
                              "nsteps_block": 10000, "seed": 42}
    assert simulation.t_start == 0.
    assert simulation.t_stop == 1000.
    assert simulation.dt == 0.01

    assert simulation.simulator == "my_simulator"
    assert simulation.conditions == {"celsius": 34.0, "v_init": -80, "other": "something"}

    assert simulation.node_sets == {"Layer23": {"layer": [2, 3]}}
    assert isinstance(simulation.spikes, SpikeReport)
    assert isinstance(simulation.spikes["default"], PopulationSpikeReport)

    assert sorted(list(simulation.reports)) == sorted(list(['soma_report', 'section_report']))
    assert isinstance(simulation.reports['soma_report'], SomaReport)
    assert isinstance(simulation.reports['section_report'], SectionReport)

    rep = simulation.reports['soma_report']
    assert sorted(list(rep.population_names)) == ["default", "default2"]
    assert isinstance(rep['default'], PopulationSomaReport)

    rep = simulation.reports['section_report']
    assert sorted(list(rep.population_names)) == ["default", "default2"]
    assert isinstance(rep['default'], PopulationSectionReport)


def test_unknonw_report():
    simulation = test_module.Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
    simulation.config["reports"] = {
        "soma_report": {
            "cells": "Layer23",
            "variable_name": "m",
            "sections": "unknown"
        }
    }

    with pytest.raises(BluepySnapError):
        simulation.reports
