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

from cached_property import cached_property

from bluepysnap.config import Config
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.node_sets import NodeSets


def _collect_frame_reports(sim):
    """Collect the different frame reports."""
    res = {}
    for name, report in sim.config["reports"].items():
        report_type = report.get("sections", "soma")
        if report_type == "soma":
            from bluepysnap.frame_report import SomaReport

            cls = SomaReport
        else:
            from bluepysnap.frame_report import CompartmentReport

            cls = CompartmentReport

        res[name] = cls(sim, name)
    return res


class Simulation:
    """Access to Simulation data."""

    def __init__(self, config):
        """Initializes a simulation object from a SONATA simulation config file.

        Args:
            config (str): Path to a SONATA simulation config file.

        Returns:
            Simulation: A Simulation object.
        """
        self._config = Config.from_simulation_config(config)

    @property
    def config(self):
        """Simulation config dictionary."""
        return self._config.to_dict()

    @cached_property
    def circuit(self):
        """Access to the circuit used for the simulation."""
        from bluepysnap.circuit import Circuit

        if "network" not in self.config:
            raise BluepySnapError("No 'network' set in the simulation/global config file.")
        return Circuit(self.config["network"])

    @property
    def run(self):
        """Access to the complete run dictionary for this simulation."""
        return self.config["run"]

    @property
    def time_start(self):
        """Returns the starting time of the simulation."""
        return self.run.get("tstart", 0)

    @property
    def time_stop(self):
        """Returns the stopping time of the simulation."""
        return self.run["tstop"]

    @property
    def dt(self):
        """Returns the frequency of reporting in milliseconds."""
        return self.run.get("dt", None)

    @property
    def time_units(self):
        """Returns the times unit for this simulation."""
        # Assuming ms at the simulation level
        return "ms"

    @property
    def conditions(self):
        """Access to the conditions dictionary for this simulation."""
        return self.config.get("conditions", {})

    @property
    def simulator(self):
        """Returns the targeted simulator."""
        return self.config.get("target_simulator")

    @cached_property
    def node_sets(self):
        """Returns the NodeSets object bound to the simulation."""
        if "node_sets_file" in self.config:
            return NodeSets(self.config["node_sets_file"])
        return {}

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
