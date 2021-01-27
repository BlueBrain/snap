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
"""Spike report access."""

from contextlib import contextmanager

from pathlib import Path
from cached_property import cached_property
import pandas as pd
import numpy as np
from libsonata import SpikeReader, SonataError

from bluepysnap.exceptions import BluepySnapError
from bluepysnap.utils import IDS_DTYPE
import bluepysnap._plotting


def _get_reader(spike_report):
    path = str(Path(spike_report.config["output_dir"]) / spike_report.config["spikes_file"])
    return SpikeReader(path)


def _collect_spikes(spike_report):
    return {population: PopulationSpikeReport(spike_report, population) for population in
            spike_report.population_names}


class PopulationSpikeReport:
    """Access to PopulationSpikeReport data."""

    def __init__(self, spike_report, population_name):
        """Initializes a PopulationSpikeReport object from a SpikeReport.

        Args:
            spike_report (SpikeReport): SpikeReport containing this spike report population.
            population_name (str): the population name corresponding to this report.

        Returns:
            PopulationSpikeReport: A PopulationSpikeReport object.
        """
        self.spike_report = spike_report
        self._spike_population = _get_reader(self.spike_report)[population_name]
        self._population_name = population_name

    @property
    def _sorted_by(self):
        """Access to the sorting attribute.

        Returns:
            str: the type of sorting used for this spike report.

            Returned values are:

               - ``'by_id'`` if the report is sorted by population node_ids
               - ``'by_time'`` if the report is sorted by timestamps
               - ``'none'`` if not sorted.
        """
        return self._spike_population.sorting

    @property
    def name(self):
        """Return the name of the population."""
        return self._population_name

    @cached_property
    def nodes(self):
        """Return the NodePopulation corresponding to this spike report."""
        return self.spike_report.simulation.circuit.nodes[self._population_name]

    def _resolve_nodes(self, group):
        """Transform a node group into a node_id array."""
        return self.nodes.ids(group=group)

    def get(self, group=None, t_start=None, t_stop=None):
        """Fetch spikes from the report.

        Args:
            group (None/int/list/np.array/dict): Get spikes filtered by group. See NodePopulation.
            t_start (float): Include only spikes occurring at or after this time.
            t_stop (float): Include only spikes occurring at or before this time.

        Returns:
            pandas.Series: return spiking node_ids indexed by sorted spike time.
        """
        node_ids = self._resolve_nodes(group).tolist()

        series_name = "ids"
        try:
            res = self._spike_population.get(node_ids=node_ids, tstart=t_start, tstop=t_stop)
        except SonataError as e:
            raise BluepySnapError(e)

        if not res:
            return pd.Series(data=[], index=pd.Index([], name="times"),
                             name=series_name, dtype=np.float64)

        res = pd.DataFrame(data=res, columns=[series_name, "times"]).set_index("times")[series_name]
        if self._sorted_by != "by_time":
            res.sort_index(inplace=True)
        return res.astype(IDS_DTYPE)

    @cached_property
    def node_ids(self):
        """Returns the node ids present in the report.

        Returns:
            np.Array: Numpy array containing the node_ids included in the report
        """
        return np.unique(self.get())


class FilteredSpikeReport:
    """Access to filtered SpikeReport data."""

    def __init__(self, spike_report, group=None, t_start=None, t_stop=None):
        """Initialize a FilteredSpikeReport.

        A FilteredSpikeReport is a lazy and cached object which contains the filtered data
        from all the populations of a report.

        Args:
            spike_report (SpikeReport): The SpikeReport to filter.
            group (None/int/list/np.array/dict): Get spikes filtered by group. See NodePopulation.
            t_start (float): Include only frames occurring at or after this time.
            t_stop (float): Include only frames occurring at or before this time.

        Returns:
            FilteredSpikeReport: A FilteredFrameReport object.
        """
        self.spike_report = spike_report
        self.group = group
        self.t_start = t_start
        self.t_stop = t_stop

    @cached_property
    def report(self):
        """Access to the report data.

        Returns:
            pandas.DataFrame: A DataFrame containing the data from the report. Row's indices are the
                different timestamps and the columns are ids and population names.
        """
        res = pd.DataFrame()
        for population in self.spike_report.population_names:
            spikes = self.spike_report[population]
            try:
                ids = spikes.nodes.ids(group=self.group)
            except BluepySnapError:
                continue
            data = spikes.get(group=ids, t_start=self.t_start, t_stop=self.t_stop).to_frame()
            data["population"] = np.full(len(data), population)
            res = pd.concat([res, data])
        if not res.empty:
            res["population"] = res["population"].astype("category")
        return res.sort_index()

    # pylint: disable=protected-access
    raster = bluepysnap._plotting.spike_raster
    firing_rate_histogram = bluepysnap._plotting.spikes_firing_rate_histogram
    isi = bluepysnap._plotting.spikes_isi
    firing_animation = bluepysnap._plotting.spikes_firing_animation


class SpikeReport:
    """Access to SpikeReport data."""

    def __init__(self, simulation):
        """Initializes a SpikeReport object from a simulation object.

        Args:
            simulation (Simulation): Simulation containing this spike report.

        Returns:
            SpikeReport: A SpikeReport object.
        """
        self._simulation = simulation

    @property
    def config(self):
        """Access to the spike 'output' config part."""
        return self._simulation.config["output"]

    @property
    def time_start(self):
        """Returns the starting time of the simulation. Default is zero."""
        return self._simulation.time_start

    @property
    def time_stop(self):
        """Returns the stopping time of the simulation."""
        return self._simulation.time_stop

    @property
    def dt(self):
        """Returns the frequency of reporting in milliseconds."""
        return self._simulation.dt

    @property
    def time_units(self):
        """Returns the time unit of reporting."""
        raise NotImplementedError

    @property
    def simulation(self):
        """Return the Simulation object related to this spike report."""
        return self._simulation

    @contextmanager
    def log(self):
        """Context manager for the spike log file."""
        path = Path(self.config["output_dir"]) / self.config["log_file"]
        if not path.exists():
            raise BluepySnapError("Cannot find the log file for the spike report.")
        yield open(str(path), "r")

    @cached_property
    def _spike_reader(self):
        """Access to the libsonata SpikeReader."""
        return _get_reader(self)

    @cached_property
    def population_names(self):
        """Returns the population names included in this report."""
        return sorted(self._spike_reader.get_population_names())

    @cached_property
    def _population(self):
        """Collect the different PopulationSpikeReports."""
        return _collect_spikes(self)

    def __getitem__(self, population_name):
        """Access the PopulationSpikeReport corresponding to the population 'population_name'."""
        return self._population[population_name]

    def __iter__(self):
        """Allows iteration over the different PopulationSpikeReports."""
        return iter(self._population)

    def filter(self, group=None, t_start=None, t_stop=None):
        """Returns a FilteredSpikeReport.

        A FilteredSpikeReport is a lazy and cached object which contains the filtered data
        from all the populations of a report.

        Args:
            group (None/int/list/np.array/dict): Get spikes filtered by group. See NodePopulation.
            t_start (float): Include only frames occurring at or after this time.
            t_stop (float): Include only frames occurring at or before this time.

        Returns:
            FilteredSpikeReport: A FilteredFrameReport object.
        """
        return FilteredSpikeReport(self, group, t_start, t_stop)
