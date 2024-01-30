# Copyright (c) 2020, EPFL/Blue Brain Project

# This file is part of BlueBrain SNAP library <https://github.com/BlueBrain/snap>

# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License version 3.0 as published
# by the Free Software Foundation.

# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.

# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""Simulation access."""

import warnings
from pathlib import Path

from cached_property import cached_property

from bluepysnap.config import SimulationConfig
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.input import get_simulation_inputs
from bluepysnap.node_sets import NodeSets


def _collect_frame_reports(sim):
    """Collect the different frame reports."""
    res = {}
    for name in sim.to_libsonata.list_report_names:
        report = sim.to_libsonata.report(name)
        report_type = report.sections.name
        if report_type == "all" or report.type.name == "lfp":
            from bluepysnap.frame_report import CompartmentReport

            cls = CompartmentReport
        elif report_type == "soma":
            from bluepysnap.frame_report import SomaReport

            cls = SomaReport
        else:
            raise BluepySnapError(f"Report {name}: format {report_type} not yet supported.")

        res[name] = cls(sim, name)
    return res


def _warn_on_overwritten_node_sets(overwritten, print_max=10):
    """Helper function to warn and print overwritten nodesets."""
    if (n := len(overwritten)) > 0:
        names = ", ".join(list(overwritten)[:print_max]) + (", ..." if n > print_max else "")
        warnings.warn(
            f"Simulation node sets overwrite {n} node set(s) in Circuit node sets: {names}",
            RuntimeWarning,
        )


class Simulation:
    """Access to Simulation data."""

    def __init__(self, config):
        """Initializes a simulation object from a SONATA simulation config file.

        Args:
            config (str): Path to a SONATA simulation config file.

        Returns:
            Simulation: A Simulation object.
        """
        self._simulation_config_path = str(Path(config).absolute())
        self._config = SimulationConfig.from_config(config)

    @property
    def to_libsonata(self):
        """Libsonata instance of the config."""
        return self._config.to_libsonata

    @property
    def config(self):
        """Simulation config dictionary."""
        return self._config.to_dict()

    @cached_property
    def circuit(self):
        """Access to the circuit used for the simulation."""
        from bluepysnap.circuit import Circuit

        if not Path(self.to_libsonata.network).is_file():
            raise BluepySnapError(f"'network' file not found: {self.to_libsonata.network}")
        return Circuit(self.to_libsonata.network)

    @property
    def output(self):
        """Access the output section."""
        return self.to_libsonata.output

    @property
    def inputs(self):
        """Access the inputs section."""
        return get_simulation_inputs(self.to_libsonata)

    @property
    def run(self):
        """Access to the complete run dictionary for this simulation."""
        return self.to_libsonata.run

    @property
    def time_start(self):
        """Returns the starting time of the simulation."""
        return 0

    @property
    def time_stop(self):
        """Returns the stopping time of the simulation."""
        return self.run.tstop

    @property
    def dt(self):
        """Returns the frequency of reporting in milliseconds."""
        return self.run.dt

    @property
    def time_units(self):
        """Returns the times unit for this simulation."""
        # Assuming ms at the simulation level
        return "ms"

    @property
    def conditions(self):
        """Access to the conditions dictionary for this simulation."""
        return getattr(self.to_libsonata, "conditions", None)

    @property
    def simulator(self):
        """Returns the targeted simulator."""
        target_simulator = getattr(self.to_libsonata, "target_simulator", None)
        return target_simulator.name if target_simulator is not None else None

    @cached_property
    def node_sets(self):
        """Returns the NodeSets object bound to the simulation."""
        try:
            path = self.circuit.to_libsonata.node_sets_path
        except BluepySnapError:  # Error raised if circuit can not be instantiated
            path = ""

        node_sets = NodeSets.from_file(path) if path else NodeSets.from_dict({})

        if self.to_libsonata.node_sets_file != path:
            overwritten = node_sets.update(NodeSets.from_file(self.to_libsonata.node_sets_file))
            _warn_on_overwritten_node_sets(overwritten)

        return node_sets

    @cached_property
    def spikes(self):
        """Access to the SpikeReport."""
        from bluepysnap.spike_report import SpikeReport

        return SpikeReport(self)

    @cached_property
    def reports(self):
        """Access all available FrameReports.

        Notes:
            Supported FrameReports are soma and compartment reports.
        """
        return _collect_frame_reports(self)

    def __getstate__(self):
        """Make Simulations pickle-able, without storing state of caches."""
        return self._simulation_config_path

    def __setstate__(self, state):
        """Load from pickle state."""
        self.__init__(state)
