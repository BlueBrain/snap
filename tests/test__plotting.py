import sys

import mock
import pytest

import bluepysnap._plotting as test_module
from bluepysnap.spike_report import SpikeReport
from bluepysnap.frame_report import FrameReport
from bluepysnap.simulation import Simulation


from utils import TEST_DATA_DIR


def test__get_pyplot():
    with mock.patch.dict(sys.modules, {'matplotlib.pyplot': None}):
        with pytest.raises(ImportError):
            test_module._get_pyplot()

    import matplotlib.pyplot
    plt_test = test_module._get_pyplot()
    assert plt_test is matplotlib.pyplot


def test__get_spike_report():
    simulation = Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
    report = SpikeReport(simulation)
    pop_report = report["default"]
    obj, pop_names = test_module._get_spike_report(report)
    assert isinstance(obj, SpikeReport)
    assert pop_names == sorted(list(report))

    obj, pop_names = test_module._get_spike_report(pop_report)
    assert isinstance(obj, SpikeReport)
    assert pop_names == [pop_report.name]


def test__get_frame_report():
    simulation = Simulation(str(TEST_DATA_DIR / 'simulation_config.json'))
    report = FrameReport(simulation, "soma_report")
    pop_report = report["default"]
    obj, pop_names = test_module._get_frame_report(report)
    assert isinstance(obj, FrameReport)
    assert pop_names == sorted(list(report))

    obj, pop_names = test_module._get_frame_report(pop_report)
    assert isinstance(obj, FrameReport)
    assert pop_names == [pop_report.name]
