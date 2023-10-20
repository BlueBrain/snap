import sys
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

import bluepysnap._plotting as test_module
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.simulation import Simulation
from bluepysnap.spike_report import FilteredSpikeReport, SpikeReport

from utils import TEST_DATA_DIR

# NOTE: The tests here are primarily to make sure all the code is covered and deprecation warnings,
# etc. are raised. They don't ensure nor really test the correctness of the functionality.


def test__get_pyplot():
    with patch.dict(sys.modules, {"matplotlib.pyplot": None}):
        with pytest.raises(ImportError):
            test_module._get_pyplot()

    import matplotlib.pyplot

    plt_test = test_module._get_pyplot()
    assert plt_test is matplotlib.pyplot


def _get_filtered_spike_report():
    return Simulation(TEST_DATA_DIR / "simulation_config.json").spikes.filter()


def _get_filtered_frame_report():
    return Simulation(TEST_DATA_DIR / "simulation_config.json").reports["soma_report"].filter()


def test_spikes_firing_rate_histogram():
    with pytest.raises(BluepySnapError, match="Invalid time_binsize"):
        test_module.spikes_firing_rate_histogram(filtered_report=None, time_binsize=0)

    filtered_report = _get_filtered_spike_report()
    ax = test_module.spikes_firing_rate_histogram(filtered_report)
    assert ax.xaxis.label.get_text() == "Time [ms]"
    assert ax.yaxis.label.get_text() == "PSTH [Hz]"

    ax.xaxis.label.set_text("Fake X")
    ax.yaxis.label.set_text("Fake Y")

    ax = test_module.spikes_firing_rate_histogram(filtered_report, ax=ax)
    assert ax.xaxis.label.get_text() == "Fake X"
    assert ax.yaxis.label.get_text() == "Fake Y"


def test_spike_raster():
    filtered_report = _get_filtered_spike_report()

    test_module.spike_raster(filtered_report)
    test_module.spike_raster(filtered_report, y_axis="y")

    ax = test_module.spike_raster(filtered_report, y_axis="mtype")

    assert ax.xaxis.label.get_text() == "Time [ms]"
    assert ax.yaxis.label.get_text() == "mtype"

    ax.xaxis.label.set_text("Fake X")
    ax.yaxis.label.set_text("Fake Y")

    ax = test_module.spike_raster(filtered_report, y_axis="mtype", ax=ax)
    assert ax.xaxis.label.get_text() == "Fake X"
    assert ax.yaxis.label.get_text() == "Fake Y"

    # Have error raised in node_population get
    filtered_report.spike_report["default"].nodes.get = Mock(
        side_effect=BluepySnapError("Fake error")
    )
    test_module.spike_raster(filtered_report, y_axis="mtype")


def test_spikes_isi():
    with pytest.raises(BluepySnapError, match="Invalid binsize"):
        test_module.spikes_isi(filtered_report=None, binsize=0)

    filtered_report = _get_filtered_spike_report()

    ax = test_module.spikes_isi(filtered_report)
    assert ax.xaxis.label.get_text() == "Interspike interval [ms]"
    assert ax.yaxis.label.get_text() == "Bin weight"

    ax = test_module.spikes_isi(filtered_report, use_frequency=True, binsize=42)
    assert ax.xaxis.label.get_text() == "Frequency [Hz]"
    assert ax.yaxis.label.get_text() == "Bin weight"

    ax.xaxis.label.set_text("Fake X")
    ax.yaxis.label.set_text("Fake Y")
    ax = test_module.spikes_isi(filtered_report, use_frequency=True, binsize=42, ax=ax)
    assert ax.xaxis.label.get_text() == "Fake X"
    assert ax.yaxis.label.get_text() == "Fake Y"

    with patch.object(test_module.np, "concatenate", Mock(return_value=[])):
        with pytest.raises(BluepySnapError, match="No data to display"):
            test_module.spikes_isi(filtered_report)


def test_spikes_firing_animation(tmp_path):
    with pytest.raises(BluepySnapError, match="Fake is not a valid axis"):
        test_module.spikes_firing_animation(filtered_report=None, x_axis="Fake")

    with pytest.raises(BluepySnapError, match="Fake is not a valid axis"):
        test_module.spikes_firing_animation(filtered_report=None, y_axis="Fake")

    filtered_report = _get_filtered_spike_report()
    anim, ax = test_module.spikes_firing_animation(filtered_report, dt=0.2)
    assert ax.title.get_text() == "time = 0.1ms"

    # convert to video to have `update_animation` called
    anim.save(tmp_path / "test.gif")

    ax.title.set_text("Fake Title")
    anim, ax = test_module.spikes_firing_animation(filtered_report, dt=0.2, ax=ax)
    assert ax.title.get_text() == "Fake Title"
    anim.save(tmp_path / "test.gif")

    # Have error raised in node_population get
    filtered_report.spike_report["default"].nodes.get = Mock(
        side_effect=BluepySnapError("Fake error")
    )

    anim, _ = test_module.spikes_firing_animation(filtered_report, dt=0.2)
    anim.save(tmp_path / "test.gif")


def test_frame_trace():
    with pytest.raises(BluepySnapError, match="Unknown plot_type Fake."):
        test_module.frame_trace(filtered_report=None, plot_type="Fake", ax="also fake")

    filtered_report = _get_filtered_frame_report()
    test_module.frame_trace(filtered_report)
    ax = test_module.frame_trace(filtered_report, plot_type="all")

    assert ax.xaxis.label.get_text() == "Time [ms]"
    assert ax.yaxis.label.get_text() == "Voltage [mV]"

    ax.xaxis.label.set_text("Fake X")
    ax.yaxis.label.set_text("Fake Y")

    ax = test_module.frame_trace(filtered_report, plot_type="all", ax=ax)

    assert ax.xaxis.label.get_text() == "Fake X"
    assert ax.yaxis.label.get_text() == "Fake Y"
