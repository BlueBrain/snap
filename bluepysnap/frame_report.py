# Copyright (c) 2019, EPFL/Blue Brain Project

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
"""Frame report access."""

from cached_property import cached_property

from pathlib2 import Path
import pandas as pd

from bluepysnap.exceptions import BluepySnapError


FORMAT_TO_EXT = {"ASCII": ".txt", "HDF5": ".h5", "BIN": ".bbp"}


def _collect_reports(frame_report, cls):
    result = {}
    for population in frame_report.population_names:
        result[population] = cls(frame_report, population)
    return result


def _get_reader(reader_report, cls):
    path = reader_report.sim.config["output"]["output_dir"]
    ext = FORMAT_TO_EXT[reader_report.config.get("format", "HDF5")]
    file_name = reader_report.config.get("file_name", reader_report.name) + ext
    path = str(Path(path, file_name))
    return cls(path)


class PopulationFrameReport(object):
    """Access to PopulationFrameReport data."""

    def __init__(self, frame_report, population_name):
        """Initializes a PopulationSpikeReport object from a SpikeReport.

        Args:
            frame_report (FrameReport): FrameReport containing this spike report population.
            population_name (str): the population name corresponding to this report.

        Returns:
            PopulationFrameReport: A PopulationFrameReport object.
        """

        self._frame_report = frame_report
        self._frame_population = self._get_reader(frame_report, population_name)
        self._population_name = population_name

    @staticmethod
    def _get_reader(frame_report, population_name):
        """Allow overriding of the Reader."""
        from libsonata import ReportReader
        return _get_reader(frame_report, ReportReader)[population_name]

    @property
    def name(self):
        """Access to the population name."""
        return self._population_name

    @property
    def sorted(self):
        """Access to the sorted attribute."""
        return self._frame_population.sorted()

    @property
    def times(self):
        """Access to the times attribute. Returns (tstart, tend, tstep) of the population."""
        return self._frame_population.times()

    @property
    def time_units(self):
        """Returns the times unit for this simulation."""
        return self._frame_population.timeUnits()

    @property
    def data_units(self):
        """Returns the data unit for this simulation."""
        return self._frame_population.dataUnits()

    @cached_property
    def nodes(self):
        """Returns the NodePopulation corresponding to this spike report."""
        result = self._frame_report.sim.circuit.nodes.get(self._population_name)
        if result is None:
            raise BluepySnapError("Undefined node population: '%s'" % self._population_name)
        return result

    def _resolve_nodes(self, group):
        """Transform a node group into a node_id array."""
        return self.nodes.ids(group=group)

    def get(self, group=None, t_start=None, t_stop=None):
        """Fetch data from the report.

        If `node_ids` is provided, filter by node_ids.
        If `t_start` and/or `t_end` is provided, filter by spike time.

        Returns:
            pandas.DataFrame: with timestamps as index and frame as columns.
        """
        node_ids = [] if group is None else self._resolve_nodes(group).tolist()
        t_start = -1 if t_start is None else t_start
        t_stop = -1 if t_stop is None else t_stop

        res = self._frame_population.get(node_ids=node_ids, tstart=t_start, tend=t_stop)
        return pd.DataFrame(data=res.data, index=res.index)


class FrameReport(object):
    """Access to FrameReport data."""

    def __init__(self, sim, report_name):
        """Initializes a FrameReport object from a simulation object.

        Args:
            sim (Simulation): Simulation containing this frame report.

        Returns:
            FrameReport: A SpikeReport object.
        """
        self._sim = sim
        self.name = report_name

    @property
    def config(self):
        """Access to the report config part."""
        return self._sim.config["reports"][self.name]

    @property
    def t_start(self):
        """Returns the starting time of the report. Default is zero."""
        return self.config.get("start_time", self._sim.t_start)

    @property
    def t_stop(self):
        """Returns the stopping time of the report."""
        return self.config.get("end_time", self._sim.t_stop)

    @property
    def dt(self):
        """Returns the frequency of reporting in milliseconds."""
        return self.config.get("dt", self._sim.dt)

    @property
    def node_set(self):
        """Returns the frequency of reporting in milliseconds."""
        return self.sim.node_sets[self.config["cells"]]

    @property
    def sim(self):
        """Return the Simulation object related to this frame report."""
        return self._sim

    @cached_property
    def _frame_reader(self):
        from libsonata import ReportReader
        return _get_reader(self, ReportReader)

    @cached_property
    def population_names(self):
        """Returns the population names included in this report."""
        return self._frame_reader.getPopulationsNames()

    @cached_property
    def _population(self):
        """Collect the different PopulationFrameReport."""
        return _collect_reports(self, PopulationFrameReport)

    def __getitem__(self, population_name):
        """Access the PopulationFrameReports corresponding to the population 'population_name'."""
        return self._population[population_name]

    def __iter__(self):
        """Allows iteration over the different PopulationFrameReports."""
        return self._population.__iter__()


class PopulationSomaReport(PopulationFrameReport):
    """Access to PopulationSomaReport data."""
    @staticmethod
    def _get_reader(frame_report, population_name):
        """Allow overriding of the Reader."""
        from libsonata import ReportReader as SomaReader
        return _get_reader(frame_report, SomaReader)[population_name]


class SomaReport(FrameReport):
    """Access to a SomaReport data """

    @cached_property
    def _frame_reader(self):
        """Override of the frame reader."""
        from libsonata import ReportReader as SomaReader
        return _get_reader(self, SomaReader)

    @cached_property
    def _population(self):
        """Override of the population collection."""
        return _collect_reports(self, PopulationSomaReport)
