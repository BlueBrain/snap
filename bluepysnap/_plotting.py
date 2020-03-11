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
"""Plotting module for the different snap objects."""
import numpy as np
import pandas as pd

from bluepysnap.exceptions import BluepySnapError


def _get_matplotlib():
    try:
        import matplotlib
        import matplotlib.pyplot as plt

    except ImportError as e:
        msg = (
            "Bluepysnap requirements are not installed.\n"
            "Please pip install as follows:\n"
            "  pip install bluepysnap[plots] --upgrade"
        )
        raise ImportError(str(e) + "\n\n" + msg)
    return matplotlib, plt


def firing_rate_histogram(spikes, group=None, t_start=None,
                          t_stop=None, binsize=None, ax=None):  #  pragma: no cover
    """PopulationSpikeReport firing rate histogram.

    Args:
        group (int/list/np.array/dict): Get spikes filtered by group. See NodePopulation.
        t_start (float): Include only spikes occurring after this time.
        t_stop (float): Include only spikes occurring before this time.
        binsize(int): bin size (milliseconds)
        ax(matplotlib.Axis): matplotlib Axis to draw on (if not specified, pyplot.gca() is used).

    Returns:
        matplotlib.Axis: Axis containing firing rate histogram.

    Notes:
        If no axis is provided through the ax=ax keyword argument,
        then a default layout is set using pyplot.gca().
    """
    _, plt = _get_matplotlib()
    data = spikes.get(group=group, t_start=t_start, t_stop=t_stop)
    if isinstance(data, np.ndarray):
        times = data
        node_count = 1
    elif isinstance(data, pd.Series):
        times = data.index.to_numpy()
        node_count = data.nunique()
    else:
        raise BluepySnapError("Wrong input type must come from PopulationSpikeReport.get().")

    time_start = np.min(times)
    time_stop = np.max(times)

    if binsize is None:
        # heuristic for a nice bin size (~100 spikes per bin on average)
        binsize = min(50.0, (time_stop - time_start) / ((len(times) / 100.) + 1.))

    bins = np.append(np.arange(time_start, time_stop, binsize), time_stop)
    hist, bin_edges = np.histogram(times, bins=bins)
    freq = 1.0 * hist / node_count / (0.001 * binsize)

    if ax is None:
        ax = plt.gca()
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('PSTH [Hz]')

    ax.plot(0.5 * (bin_edges[1:] + bin_edges[:-1]), freq, label="PSTH", drawstyle='steps-mid')
    return ax


def raster(spikes, group=None, t_start=None, t_stop=None, ax=None):  # pragma: no cover
    """PopulationSpikeReport raster plot.

    Args:
        group (int/list/np.array/dict): Get spikes filtered by group. See NodePopulation.
        t_start (float): Include only spikes occurring after this time.
        t_stop (float): Include only spikes occurring before this time.
        ax(matplotlib.Axis): matplotlib Axis to draw on (if not specified, pyplot.gca() is used).

    Returns:
        matplotlib.Axis: Axis containing Spikes raster plot.

    Notes:
        If no axis is provided through the ax=ax keyword argument,
        then a default layout is set using pyplot.gca().
    """
    _, plt = _get_matplotlib()

    time_start, time_stop = spikes.spike_report.time_start, spikes.spike_report.time_stop

    if ax is None:
        ax = plt.gca()
        ax.xaxis.grid()
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("{} GIDs".format(spikes.name))
        ax.tick_params(axis='y', which='both', length=0)
        ax.set_xlim(time_start, time_stop)
        ax.set_ylim(0, spikes.nodes.size)

    data = spikes.get(group=group, t_start=t_start, t_stop=t_stop)
    ts = data.index
    ys = data.to_numpy()
    ax.scatter(ts, ys, s=10, marker='|')
    return ax


def isi(spikes, group=None, t_start=None, t_stop=None,
        freq=False, binsize=None, ax=None):  # pragma: no cover
    # pylint: disable=too-many-arguments
    """
    Interspike interval histogram.

    Args:
        group (int/list/np.array/dict): Get spikes filtered by group. See NodePopulation.
        t_start (float): Include only spikes occurring after this time.
        t_stop (float): Include only spikes occurring before this time.
        ax(matplotlib.Axis): matplotlib Axis to draw on (if not specified, pyplot.gca() is used.

    Returns:
        matplotlib Axes with interspike interval histogram.

    Notes:
        If no axis is provided through the ax=ax keyword argument,
        then a default layout is set using pyplot.gca().
    """
    _, plt = _get_matplotlib()

    spikes = spikes.spikes.get(group=group, t_start=t_start, t_stop=t_stop)

    values = [np.diff(gid_spikes.index.values)
              for _, gid_spikes in spikes.groupby(spikes)]
    if values:
        values = np.concatenate(values)
    else:
        values = np.array(values)

    if freq:
        values = values[values > 0]  # filter out zero intervals (well, you never know)
        values = 1000.0 / values

    if binsize is None:
        bins = 'auto'
    else:
        bins = np.arange(0, np.max(values), binsize)

    if ax is None:
        ax = plt.gca()
        if freq:
            ax.set_xlabel('Frequency [Hz]')
        else:
            ax.set_xlabel('Interspike interval [ms]')
        ax.set_ylabel('Bin weight')

    ax.hist(values, bins=bins, edgecolor='black', density=True, label=label)
    return ax