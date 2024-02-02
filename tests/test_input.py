import libsonata
import numpy.testing as npt
import pytest

import bluepysnap.input as test_module
from bluepysnap.exceptions import BluepySnapError

from utils import TEST_DATA_DIR


class TestSynapseReplay:
    def setup_method(self):
        simulation = libsonata.SimulationConfig.from_file(TEST_DATA_DIR / "simulation_config.json")
        self.test_obj = test_module.SynapseReplay(simulation.input("spikes_1"))

    def test_all(self):
        snap_attrs = {a for a in dir(self.test_obj) if not a.startswith("_")}
        libsonata_attrs = {a for a in dir(self.test_obj._instance) if not a.startswith("_")}

        # check that wrapped instance's public methods are available in the object
        assert snap_attrs.symmetric_difference(libsonata_attrs) == {"reader"}
        assert isinstance(self.test_obj.reader, libsonata.SpikeReader)

        for a in libsonata_attrs:
            assert getattr(self.test_obj, a) == getattr(self.test_obj._instance, a)

        npt.assert_almost_equal(self.test_obj.reader["default"].get(), [[0, 10.775]])

    def test_no_such_attribute(self):
        """Check that the attribute error is raised from the wrapped libsonata object."""
        with pytest.raises(AttributeError, match="libsonata._libsonata.SynapseReplay"):
            self.test_obj.no_such_attribute


def test_get_simulation_inputs():
    simulation = libsonata.SimulationConfig.from_file(TEST_DATA_DIR / "simulation_config.json")
    inputs = test_module.get_simulation_inputs(simulation)

    assert isinstance(inputs, dict)
    assert inputs.keys() == {"spikes_1", "current_clamp_1"}

    assert isinstance(inputs["spikes_1"], test_module.SynapseReplay)

    try:
        Linear = libsonata._libsonata.Linear
    except AttributeError:
        from libsonata._libsonata import SimulationConfig

        Linear = SimulationConfig.Linear

    assert isinstance(inputs["current_clamp_1"], Linear)

    with pytest.raises(BluepySnapError, match="Unexpected type for 'simulation': str"):
        test_module.get_simulation_inputs("fail_me")
