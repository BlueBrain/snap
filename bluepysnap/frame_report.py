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
from collections.abc import Mapping

import numpy as np
import pandas as pd
from cached_property import cached_property
from libsonata import ElementReportReader, SonataError

import bluepysnap._plotting
from bluepysnap import query
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.utils import ensure_ids

L = logging.getLogger(__name__)


def _collect_population_reports(frame_report, cls):
    return {
        population: cls(frame_report, population) for population in frame_report.population_names
    }


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
        return ElementReportReader(frame_report.to_libsonata.file_name)[population_name]

    @property
    def name(self):
        """Access to the population name."""
        return self._population_name

    def resolve_nodes(self, group, raise_missing_property=True):
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

    def get(self, group=None, t_start=None, t_stop=None, t_step=None):
        """Fetch data from the report.

        Args:
            group (None/int/list/np.array/dict): Get frames filtered by :ref:`Group Concept`.
            t_start (float): Include only frames occurring at or after this time.
            t_stop (float): Include only frames occurring at or before this time.
            t_step (float): Optional time step, useful to reduce the number of samples.
                It should be a multiple of the report time step dt, and it's equal to dt by default.
                If the given t_step isn't an exact multiple, it's rounded to the closer multiple.
                Only the samples at t = t0 + k * t_step, for k = 0, 1... are returned,
                where t0 is the first sample time >= t_start.

        Returns:
            pandas.DataFrame: frame as columns indexed by timestamps.
        """
        t_stride = round(t_step / self.frame_report.dt) if t_step is not None else 1
        if t_stride < 1:
            msg = f"Invalid {t_step=}. It should be None or a multiple of {self.frame_report.dt}."
            raise BluepySnapError(msg)
        ids = self.resolve_nodes(group).tolist()
        try:
            view = self._frame_population.get(
                node_ids=ids, tstart=t_start, tstop=t_stop, tstride=t_stride
            )
        except SonataError as e:
            raise BluepySnapError(e) from e

        # cell ids and section ids in the columns are enforced to be int64
        # to avoid issues with numpy automatic conversions and to ensure that
        # the results are the same regardless of the libsonata version [NSETM-1766]
        res = pd.DataFrame(
            data=view.data,
            columns=pd.MultiIndex.from_arrays(ensure_ids(view.ids).T),
            index=view.times,
        ).sort_index(axis=1)

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
            group (None/int/list/np.array/dict): Get frames filtered by :ref:`Group Concept`.
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
            different timestamps and the column's MultiIndex are:

            - (population_name, node_id, compartment id) for the CompartmentReport
            - (population_name, node_id) for the SomaReport
        """
        dataframes = {}
        for population in self.frame_report.population_names:
            frames = self.frame_report[population]
            ids = frames.resolve_nodes(self.group, raise_missing_property=False)
            df = frames.get(group=ids, t_start=self.t_start, t_stop=self.t_stop)
            dataframes[population] = df
        # optimize when there is at most one non-empty df: use copy=False, and no need to sort
        if sum(not df.empty for df in dataframes.values()) <= 1:
            return pd.concat(dataframes, axis=1, copy=False)
        # when concatenating multiple df, don't use copy=False because 2x slower (Pandas 2.0.2)
        result = pd.concat(dataframes, axis=1)
        return result.sort_index(axis=0).sort_index(axis=1)

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
        return ElementReportReader(self.to_libsonata.file_name)

    @property
    def to_libsonata(self):
        """Access libsonata instance of the report."""
        return self.simulation.to_libsonata.report(self.name)

    @property
    def config(self):
        """Access the report config."""
        return self.simulation.config["reports"][self.name]

    @property
    def time_start(self):
        """Returns the starting time of the report."""
        return self.to_libsonata.start_time

    @property
    def time_stop(self):
        """Returns the stopping time of the report."""
        return self.to_libsonata.end_time

    @property
    def dt(self):
        """Returns the frequency of reporting in milliseconds."""
        dt = self.to_libsonata.dt
        if dt != self.simulation.dt:
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
        """Returns the name of the node set for the report."""
        return self.to_libsonata.cells

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
            group (None/int/list/np.array/dict): Get frames filtered by :ref:`Group Concept`.
            t_start (float): Include only frames occurring at or after this time.
            t_stop (float): Include only frames occurring at or before this time.

        Returns:
            FilteredFrameReport: A FilteredFrameReport object.
        """
        return FilteredFrameReport(self, group, t_start, t_stop)


class PopulationCompartmentReport(PopulationFrameReport):
    """Access to PopulationCompartmentsReport data."""

    @property
    def _node_sets(self):
        """Access to simulation node sets."""
        return self.frame_report.simulation.node_sets

    @cached_property
    def nodes(self):
        """Returns the NodePopulation corresponding to this report."""
        return self.frame_report.simulation.circuit.nodes[self._population_name]

    def resolve_nodes(self, group, raise_missing_property=True):
        """Transform a group into a node_id array."""
        if isinstance(group, str):
            group = self._node_sets[group]
        elif isinstance(group, Mapping):
            group = query.resolve_nodesets(
                self._node_sets, self.nodes, group, raise_missing_property
            )
        return self.nodes.ids(group=group, raise_missing_property=raise_missing_property)


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
