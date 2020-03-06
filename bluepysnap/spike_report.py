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
"""Spike report access."""

from contextlib import contextmanager

from pathlib2 import Path
from cached_property import cached_property
import pandas as pd

from bluepysnap.exceptions import BluepySnapError


def _get_reader(spike_report):
    from libsonata import SpikeReader
    path = str(Path(spike_report.config["output_dir"]) / spike_report.config["spikes_file"])
    return SpikeReader(path)


def _collect_spikes(spike_report):
    result = {}
    for population in spike_report.population_names:
        result[population] = PopulationSpikeReport(spike_report, population)
    return result


class PopulationSpikeReport(object):
    """Access to PopulationSpikeReport data."""

    def __init__(self, spike_report, population_name):
        """Initializes a PopulationSpikeReport object from a SpikeReport.

        Args:
            spike_report (SpikeReport): SpikeReport containing this spike report population.
            population_name (str): the population name corresponding to this report.

        Returns:
            PopulationSpikeReport: A PopulationSpikeReport object.
        """
        self._spike_report = spike_report
        self._spike_population = _get_reader(self._spike_report)[population_name]
        self._population_name = population_name

    @property
    def sorting(self):
        """Access to the sorting attribute.

        Returns:
            str: the type of sorting used for this spike report.

        Notes:
            Returned values is:
            'by_id' if the report is sorted by population node_ids
            'by_time' if the report is sorted by timestamps
            'none' if not sorted.
        """
        return self._spike_population.sorting

    @property
    def name(self):
        """Return the name of the population."""
        return self._population_name

    @cached_property
    def nodes(self):
        """Returns the NodePopulation corresponding to this spike report."""
        result = self._spike_report.sim.circuit.nodes.get(self._population_name)
        if result is None:
            raise BluepySnapError("Undefined node population: '%s'" % self._population_name)
        return result

    def _resolve_nodes(self, group):
        """Transform a node group into a node_id array."""
        return self.nodes.ids(group=group)

    def get(self, group=None, t_start=None, t_stop=None):
        """Fetch spikes from the report.

        If `node_ids` is provided, filter by node_ids.
        If `t_start` and/or `t_end` is provided, filter by spike time.

        Returns:
            pandas.Series: spiking node_ids indexed by sorted spike time.
        """
        node_ids = [] if group is None else self._resolve_nodes(group).tolist()

        t_start = -1 if t_start is None else t_start
        t_stop = -1 if t_stop is None else t_stop
        series_name = "{}_node_ids".format(self._population_name)

        res = self._spike_population.get(node_ids=node_ids, tstart=t_start, tstop=t_stop)
        if not res:
            return pd.Series(data=[], index=[], name=series_name)

        node_ids, times = zip(*res)
        res = pd.Series(data=node_ids, index=times, name=series_name)
        if self.sorting == "by_time":
            return res
        return res.sort_index()

    def get_node_id(self, node_id, t_start=None, t_stop=None):
        """Fetch spikes from the report for a given `node_id`.

        If `t_start` and/or `t_end` is provided, filter by spike time.

        Returns:
            numpy.ndarray: with sorted spike times.
        """
        return self.get(node_id, t_start=t_start, t_stop=t_stop).index.to_numpy()


class SpikeReport(object):
    """Access to SpikeReport data."""

    def __init__(self, sim):
        """Initializes a SpikeReport object from a simulation object.

        Args:
            sim (Simulation): Simulation containing this spike report.

        Returns:
            SpikeReport: A SpikeReport object.
        """
        self._sim = sim

    @property
    def config(self):
        """Access to the spike 'output' config part."""
        return self._sim.config["output"]

    @property
    def t_start(self):
        """Returns the starting time of the simulation. Default is zero."""
        return self._sim.t_start

    @property
    def t_stop(self):
        """Returns the stopping time of the simulation."""
        return self._sim.t_stop

    @property
    def dt(self):
        """Returns the frequency of reporting in milliseconds."""
        return self._sim.dt

    @property
    def sim(self):
        """Return the Simulation object related to this spike report."""
        return self._sim

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
        return self._spike_reader.get_populations_names()

    @cached_property
    def _population(self):
        """Collect the different PopulationSpikeReports."""
        return _collect_spikes(self)

    def __getitem__(self, population_name):
        """Access the PopulationSpikeReport corresponding to the population 'population_name'."""
        return self._population[population_name]

    def __iter__(self):
        """Allows iteration over the different PopulationSpikeReports."""
        return self._population.__iter__()
