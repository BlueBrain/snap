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
"""Frame report access."""
from cached_property import cached_property

from pathlib2 import Path
import pandas as pd

from bluepysnap.exceptions import BluepySnapError


FORMAT_TO_EXT = {"ASCII": ".txt", "HDF5": ".h5", "BIN": ".bbp"}


def _collect_population_reports(frame_report, cls):
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
        """Initializes a PopulationFrameReport object from a FrameReport.

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
        """Access to the population compartment reader."""
        from libsonata import ElementsReportReader
        return _get_reader(frame_report, ElementsReportReader)[population_name]

    @property
    def name(self):
        """Access to the population name."""
        return self._population_name

    @property
    def sorted(self):
        """Access to the sorted attribute."""
        return self._frame_population.sorted

    @property
    def times(self):
        """Access to the times attribute. Returns (tstart, tend, tstep) of the population."""
        return self._frame_population.times

    @property
    def time_units(self):
        """Returns the times unit for this simulation."""
        return self._frame_population.time_units

    @property
    def data_units(self):
        """Returns the data unit for this simulation."""
        return self._frame_population.data_units

    @cached_property
    def population(self):
        """Returns the Population corresponding to this report.

        Notes:
            The name population is to envision the future synapse report with witch we will
            connect to a edge population (maybe).
        """
        result = self._frame_report.sim.circuit.nodes.get(self._population_name)
        if result is None:
            raise BluepySnapError("Undefined node population: '%s'" % self._population_name)
        return result

    def _resolve(self, group):
        """Transform a group into ids array."""
        return self.population.ids(group=group)

    @staticmethod
    def _wrap_columns(columns):
        """Should be overloaded for soma or other report types."""
        return columns

    def get(self, group=None, t_start=None, t_stop=None):
        """Fetch data from the report.

        If `group` is provided, filter by frame ids.
        If `t_start` and/or `t_end` is provided, filter by spike time.

        Returns:
            pandas.DataFrame: with timestamps as index and frame as columns.
        """
        ids = [] if group is None else self._resolve(group).tolist()
        t_start = -1 if t_start is None else t_start
        t_stop = -1 if t_stop is None else t_stop

        view = self._frame_population.get(node_ids=ids, tstart=t_start, tstop=t_stop)
        if not view.data:
            return pd.DataFrame()
        res = pd.DataFrame(data=view.data, index=view.index)
        # rename from multi index to index cannot be achieved easily through df.rename
        res.columns = self._wrap_columns(res.columns)
        res.sort_index(inplace=True)
        return res


class FrameReport(object):
    """Access to FrameReport data."""

    def __init__(self, sim, report_name):
        """Initializes a FrameReport object from a simulation object.

        Args:
            sim (Simulation): Simulation containing this frame report.

        Returns:
            FrameReport: A FrameReport object.
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
        """Access to the compartment report reader."""
        from libsonata import ElementsReportReader
        return _get_reader(self, ElementsReportReader)

    @cached_property
    def population_names(self):
        """Returns the population names included in this report."""
        return self._frame_reader.get_populations_names()

    @cached_property
    def _population_report(self):
        """Collect the different PopulationFrameReport."""
        return _collect_population_reports(self, PopulationFrameReport)

    def __getitem__(self, population_name):
        """Access the PopulationFrameReports corresponding to the population 'population_name'."""
        return self._population_report[population_name]

    def __iter__(self):
        """Allows iteration over the different PopulationFrameReports."""
        return self._population_report.__iter__()


class PopulationCompartmentsReport(PopulationFrameReport):
    """Access to PopulationCompartmentsReport data."""


class CompartmentsReport(FrameReport):
    """Access to a CompartmentsReport data """
    @cached_property
    def _population_report(self):
        """Collect the different PopulationCompartmentsReport."""
        return _collect_population_reports(self, PopulationCompartmentsReport)


class PopulationSomasReport(PopulationFrameReport):
    """Access to PopulationSomaReport data."""
    @staticmethod
    def _wrap_columns(columns):
        return columns.levels[0]


class SomasReport(FrameReport):
    """Access to a SomaReport data."""
    @cached_property
    def _population_report(self):
        """Collect the different PopulationSomaReport."""
        return _collect_population_reports(self, PopulationSomasReport)

