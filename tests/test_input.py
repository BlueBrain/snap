import libsonata
import numpy.testing as npt
import pytest

import bluepysnap.input as test_module

from utils import TEST_DATA_DIR


class TestSynapseReplay:
    def setup_method(self):
        simulation = libsonata.SimulationConfig.from_file(TEST_DATA_DIR / "simulation_config.json")
        self.simulation = simulation
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


class TestInput:
    def setup_method(self):
        simulation = libsonata.SimulationConfig.from_file(TEST_DATA_DIR / "simulation_config.json")
        self.test_obj = test_module.Input(simulation)
        self.as_dict = test_module.Input.as_dict(simulation)

    def test_all(self):
        input_obj = self.test_obj
        assert input_obj.keys() == {"spikes_1", "current_clamp_1"}

        assert isinstance(input_obj["spikes_1"], test_module.SynapseReplay)
        assert isinstance(input_obj["current_clamp_1"], libsonata._libsonata.Linear)

        input_dict = self.as_dict
        assert isinstance(input_dict, dict)
        assert set(input_dict) == set(input_obj.keys())

        assert isinstance(input_dict["spikes_1"], test_module.SynapseReplay)
        assert isinstance(input_dict["current_clamp_1"], libsonata._libsonata.Linear)
