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
import logging
from pathlib import Path

from cached_property import cached_property
import numpy as np
import pandas as pd
from libsonata import ElementReportReader, SonataError

import bluepysnap._plotting
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.utils import ensure_list, ensure_ids

L = logging.getLogger(__name__)

FORMAT_TO_EXT = {"ASCII": ".txt", "HDF5": ".h5", "BIN": ".bbp"}


def _collect_population_reports(frame_report, cls):
    return {population: cls(frame_report, population) for population in
            frame_report.population_names}


def _get_reader(reader_report, cls):
    path = reader_report.simulation.config["output"]["output_dir"]
    ext = FORMAT_TO_EXT[reader_report.config.get("format", "HDF5")]
    file_name = reader_report.config.get("file_name", reader_report.name) + ext
    path = str(Path(path, file_name))
    return cls(path)


class PopulationFrameReport:
    """Access to PopulationFrameReport data."""

    def __init__(self, frame_report, population_name):
        """Initializes a PopulationFrameReport object from a FrameReport.

        Args:
            frame_report (FrameReport): FrameReport containing this frame report population.
            population_name (str): the population name corresponding to this report.

        Returns:
            PopulationFrameReport: A PopulationFrameReport object.
        """
        self.frame_report = frame_report
        self._frame_population = self._get_reader(frame_report, population_name)
        self._population_name = population_name

    @staticmethod
    def _get_reader(frame_report, population_name):
        """Access to the population compartment reader."""
        return _get_reader(frame_report, ElementReportReader)[population_name]

    @property
    def name(self):
        """Access to the population name."""
        return self._population_name

    def _resolve(self, group):
        """Transform a group into ids array.

        Notes:
            The type of ids depends on the type of report and so this function needs to be
            implemented for all type of reports. It can return node or edge ids or something else.
        """
        raise NotImplementedError

    @staticmethod
    def _wrap_columns(columns):
        """Allows to change the columns names if needed."""
        return columns

    def get(self, group=None, t_start=None, t_stop=None):
        """Fetch data from the report.

        Args:
            group (None/int/list/np.array/dict): Get frames filtered by group. See NodePopulation.
            t_start (float): Include only frames occurring at or after this time.
            t_stop (float): Include only frames occurring at or before this time.

        Returns:
            pandas.DataFrame: frame as columns indexed by timestamps.
        """
        ids = self._resolve(group).tolist()
        try:
            view = self._frame_population.get(node_ids=ids, tstart=t_start, tstop=t_stop)
        except SonataError as e:
            raise BluepySnapError(e)

        if len(view.ids) == 0:
            return pd.DataFrame()

        res = pd.DataFrame(data=view.data,
                           columns=pd.MultiIndex.from_arrays(np.asarray(view.ids).T),
                           index=view.times).sort_index(axis=1)

        # rename from multi index to index cannot be achieved easily through df.rename
        res.columns = self._wrap_columns(res.columns)
        return res

    @cached_property
    def node_ids(self):
        """Returns the node ids present in the report.

        Returns:
            np.Array: Numpy array containing the node_ids included in the report
        """
        return np.sort(ensure_ids(self._frame_population.get_node_ids()))


class FilteredFrameReport:
    """Access to filtered FrameReport data."""

    def __init__(self, frame_report, group=None, t_start=None, t_stop=None):
        """Initialize a FilteredFrameReport.

        A FilteredFrameReport is a lazy and cached object which contains the filtered data
        from all the populations of a report.

        Args:
            frame_report (FrameReport): The FrameReport to filter.
            group (None/int/list/np.array/dict): Get frames filtered by group. See NodePopulation.
            t_start (float): Include only frames occurring at or after this time.
            t_stop (float): Include only frames occurring at or before this time.

        Returns:
            FilteredFrameReport: A FilteredFrameReport object.
        """
        self.frame_report = frame_report
        self.group = group
        self.t_start = t_start
        self.t_stop = t_stop

    @cached_property
    def report(self):
        """Access to the report data.

        Returns:
            pandas.DataFrame: A DataFrame containing the data from the report. Row's indices are the
                different timestamps and the column's MultiIndex are :
                - (population_name, node_id, compartment id) for the CompartmentReport
                - (population_name, node_id) for the SomaReport
        """
        res = pd.DataFrame()
        for population in self.frame_report.population_names:
            frames = self.frame_report[population]
            try:
                ids = frames.nodes.ids(group=self.group)
            except BluepySnapError:
                continue
            data = frames.get(group=ids, t_start=self.t_start, t_stop=self.t_stop)
            if data.empty:
                continue
            new_index = tuple(tuple([population] + ensure_list(x)) for x in data.columns)
            data.columns = pd.MultiIndex.from_tuples(new_index)
            # need to do this in order to preserve MultiIndex for columns
            res = data if res.empty else data.join(res, how='outer')
        return res.sort_index().sort_index(axis=1)

    # pylint: disable=protected-access
    trace = bluepysnap._plotting.frame_trace


class FrameReport:
    """Access to FrameReport data."""

    def __init__(self, simulation, report_name):
        """Initializes a FrameReport object from a simulation object.

        Args:
            simulation (Simulation): Simulation containing this frame report.
            report_name (str): The name of this frame report.

        Returns:
            FrameReport: A FrameReport object.
        """
        self._simulation = simulation
        self.name = report_name

    @cached_property
    def _frame_reader(self):
        """Access to the compartment report reader."""
        return _get_reader(self, ElementReportReader)

    @property
    def config(self):
        """Access the report config."""
        return self._simulation.config["reports"][self.name]

    @property
    def time_start(self):
        """Returns the starting time of the report."""
        return self.config.get("start_time", self._simulation.time_start)

    @property
    def time_stop(self):
        """Returns the stopping time of the report."""
        return self.config.get("end_time", self._simulation.time_stop)

    @property
    def dt(self):
        """Returns the frequency of reporting in milliseconds."""
        dt = self.config.get("dt", self._simulation.dt)
        if dt != self._simulation.dt:
            L.warning("dt from the report differs from the global simulation dt.")
        return dt

    @property
    def time_units(self):
        """Returns the data unit for this report."""
        units = {self._frame_reader[pop].time_units for pop in self.population_names}
        if len(units) > 1:
            raise BluepySnapError("Multiple time units found in the different populations.")
        return units.pop()

    @cached_property
    def data_units(self):
        """Returns the data unit for this report."""
        units = {self._frame_reader[pop].data_units for pop in self.population_names}
        if len(units) > 1:
            raise BluepySnapError("Multiple data units found in the different populations.")
        return units.pop()

    @property
    def node_set(self):
        """Returns the node set for the report."""
        return self.simulation.node_sets[self.config["cells"]]

    @property
    def simulation(self):
        """Return the Simulation object related to this frame report."""
        return self._simulation

    @cached_property
    def population_names(self):
        """Returns the population names included in this report."""
        return sorted(self._frame_reader.get_population_names())

    @cached_property
    def _population_report(self):
        """Collect the different PopulationFrameReport."""
        return _collect_population_reports(self, PopulationFrameReport)

    def __getitem__(self, population_name):
        """Access the PopulationFrameReports corresponding to the population 'population_name'."""
        return self._population_report[population_name]

    def __iter__(self):
        """Allows iteration over the different PopulationFrameReports."""
        return iter(self._population_report)

    def filter(self, group=None, t_start=None, t_stop=None):
        """Returns a FilteredFrameReport.

        A FilteredFrameReport is a lazy and cached object which contains the filtered data
        from all the populations of a report.

        Args:
            group (None/int/list/np.array/dict): Get frames filtered by group. See NodePopulation.
            t_start (float): Include only frames occurring at or after this time.
            t_stop (float): Include only frames occurring at or before this time.

        Returns:
            FilteredFrameReport: A FilteredFrameReport object.
        """
        return FilteredFrameReport(self, group, t_start, t_stop)


class PopulationCompartmentReport(PopulationFrameReport):
    """Access to PopulationCompartmentsReport data."""

    @cached_property
    def nodes(self):
        """Returns the NodePopulation corresponding to this report."""
        return self.frame_report.simulation.circuit.nodes[self._population_name]

    def _resolve(self, group):
        """Transform a group into a node_id array."""
        return self.nodes.ids(group=group)


class CompartmentReport(FrameReport):
    """Access to a CompartmentsReport data."""

    @cached_property
    def _population_report(self):
        """Collect the different PopulationCompartmentsReport."""
        return _collect_population_reports(self, PopulationCompartmentReport)


class PopulationSomaReport(PopulationCompartmentReport):
    """Access to PopulationSomaReport data."""

    @staticmethod
    def _wrap_columns(columns):
        """Transform pandas.MultiIndex into pandas.Index for the pandas.DataFrame columns.

        Notes:
            the libsonata.ElementsReader.get() returns tuple as columns for the data. For the
            soma reports it means: pandas.MultiIndex([(0, 0), (1, 0), ..., (last_node_id, 0)]).
            So we convert this into pandas.Index([0,1,..., last_node_id]).
        """
        return columns.levels[0]


class SomaReport(FrameReport):
    """Access to a SomaReport data."""

    @cached_property
    def _population_report(self):
        """Collect the different PopulationSomasReport."""
        return _collect_population_reports(self, PopulationSomaReport)
