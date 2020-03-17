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
import logging
import numpy as np

from bluepysnap.exceptions import BluepySnapError
from bluepysnap.sonata_constants import Node

L = logging.getLogger(__name__)


def _get_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        msg = (
            "Bluepysnap requirements are not installed.\n"
            "Please pip install as follows:\n"
            "  pip install bluepysnap[plots] --upgrade"
        )
        raise ImportError(str(e) + "\n\n" + msg)
    return plt


def spikes_firing_rate_histogram(spikes, group=None, t_start=None,
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
    plt = _get_pyplot()
    data = spikes.get(group=group, t_start=t_start, t_stop=t_stop)
    if isinstance(data, np.ndarray):
        times = data
        node_count = 1
    else:
        times = data.index.to_numpy()
        node_count = data.nunique()

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


def spikes_raster(spikes, group=None, t_start=None, t_stop=None, ax=None):  # pragma: no cover
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
    plt = _get_pyplot()

    if ax is None:
        ax = plt.gca()
        ax.xaxis.grid()
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("{} node ids".format(spikes.name))
        ax.tick_params(axis='y', which='both', length=0)
        time_start, time_stop = spikes.spike_report.time_start, spikes.spike_report.time_stop
        ax.set_xlim(time_start, time_stop)
        ax.set_ylim(0, spikes.nodes.size)

    data = spikes.get(group=group, t_start=t_start, t_stop=t_stop)
    ts = data.index
    ys = data.to_numpy()
    ax.scatter(ts, ys, s=10, marker='|')
    return ax


def spikes_isi(spikes, group=None, t_start=None, t_stop=None,
               use_frequency=False, binsize=None, ax=None):  # pragma: no cover
    # pylint: disable=too-many-arguments
    """PopulationSpikeReport Interspike interval histogram.

    Args:
        group (int/list/np.array/dict): Get spikes filtered by group. See NodePopulation.
        t_start (float): Include only spikes occurring after this time.
        t_stop (float): Include only spikes occurring before this time.
        use_frequency(bool): use inverse interspike interval times (Hz)
        binsize(float): bin size (milliseconds or Hz)
        ax(matplotlib.Axis): matplotlib Axis to draw on (if not specified, pyplot.gca() is used).

    Returns:
        matplotlib Axes with interspike interval histogram.

    Notes:
        If no axis is provided through the ax=ax keyword argument,
        then a default layout is set using pyplot.gca().
    """
    plt = _get_pyplot()

    data = spikes.get(group=group, t_start=t_start, t_stop=t_stop)

    values = [np.diff(node_spikes.index.values)
              for _, node_spikes in data.groupby(data)]
    if values:
        values = np.concatenate(values)
    else:
        values = np.array(values)

    if use_frequency:
        values = values[values > 0]  # filter out zero intervals
        values = 1000.0 / values

    if binsize is None:
        bins = 'auto'
    else:
        bins = np.arange(0, np.max(values), binsize)

    if ax is None:
        ax = plt.gca()
        if use_frequency:
            ax.set_xlabel('Frequency [Hz]')
        else:
            ax.set_xlabel('Interspike interval [ms]')
        ax.set_ylabel('Bin weight')

    ax.hist(values, bins=bins, edgecolor='black', density=True)
    return ax


def spikes_firing_animation(spikes, group=None, t_start=None,
                            t_stop=None, x_axis=Node.X, y_axis=Node.Y,
                            dt=20, ax=None):  # pragma: no cover
    # pylint: disable=too-many-locals,too-many-arguments,anomalous-backslash-in-string
    """PopulationSpikeReport simple animation of simulation spikes.

    Args:
        group (int/list/np.array/dict): Get spikes filtered by group. See NodePopulation.
        t_start (float): Include only spikes occurring after this time.
        t_stop (float): Include only spikes occurring before this time.
        x_axis (str): Node enum that will determine the animation x_axis
        y_axis (str): Node enum that will determine the animation y_axis
        dt (int) : the time bin size of each frame in the video in ms
        ax(matplotlib.Axis): matplotlib Axis to draw on (if not specified, pyplot.gca()
        and plt.figure() are used).

    Returns :
        (matplotlib.animation.FuncAnimation, matplotlib.Axis): the matplotlib animation object and
        the corresponding axis.

    Notes:
        From scripts:
        >>> import matplotlib.pyplot as plt
        >>> from bluepysnap import Simulation
        >>> report = Simulation("config.json").spikes["my_population"]
        >>> anim, ax = report.firing_animation()
        >>> plt.show()
        >>> # to save the animation : do not plt.show() and just anim.save('my_movie.mp4')

        From notebooks:
        >>> from IPython.display import HTML
        >>> from bluepysnap import Simulation
        >>> report = Simulation("config.json").spikes["my_population"]
        >>> anim, ax = report.firing_animation()
        >>> HTML(anim.to_html5_video())
    """
    plt = _get_pyplot()
    from matplotlib.animation import FuncAnimation

    def _check_axis(axis):
        """ Verifies axes values """
        axes = {Node.X, Node.Y, Node.Z}
        if axis not in axes:
            raise BluepySnapError('{} is not a valid axis'.format(axis))

    _check_axis(x_axis)
    _check_axis(y_axis)

    data = spikes.get(group=group, t_start=t_start, t_stop=t_stop)
    active_gids = data.unique()
    positions = spikes.nodes.get(group=active_gids, properties=[x_axis, y_axis])

    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
        ax.set_title('time = {}ms'.format(t_start))
        x_limits = [positions[x_axis].min(), positions[x_axis].max()]
        y_limits = [positions[y_axis].min(), positions[y_axis].max()]
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_xlabel('{} $\mu$m'.format(x_axis))  # noqa
        ax.set_ylabel('{} $\mu$m'.format(y_axis))  # noqa

    else:
        fig = ax.figure

    dots = ax.plot([], [], '.k')

    def update_animation(frame):
        """ Update the animation plots and axes"""
        ax.set_title('time = ' + str(frame * dt) + ' ms')
        mask = (data.index >= frame * dt) & (data.index <= (frame + 1) * dt)
        frame_gids = data[mask].unique()
        x = positions.loc[frame_gids, x_axis].values
        y = positions.loc[frame_gids, y_axis].values
        dots[0].set_data(x, y)
        return dots

    frames = list(range(int(data.index[0] / dt), int(data.index[-1] / dt)))
    anim = FuncAnimation(fig, update_animation, frames=frames)
    return anim, ax


def soma_trace(frames, group=None, t_start=None, t_stop=None, plot_type='mean', ax=None):  # pragma: no cover
    # pylint: disable=too-many-arguments
    """PopulationSomasReport potential plot displaying the voltage as a function of time.

    Args:
        group (int/list/np.array/dict): Get spikes filtered by group. See NodePopulation.
        t_start (float): Include only spikes occurring after this time.
        t_stop (float): Include only spikes occurring before this time.
        plot_type (str): string either `all` or `mean`. `all` will plot the first 15 traces from the
        group. `mean` will plot the mean value of the node
        ax: A plot axis object that will be updated
    """
    plt = _get_pyplot()

    if ax is None:
        ax = plt.gca()
        data_units = frames.frame_report.data_units
        if plot_type == "mean":
            ax.set_ylabel('Avg volt. [{}]'.format(data_units))
        elif plot_type == "all":
            ax.set_ylabel('Voltage [{}]'.format(data_units))
        ax.set_xlabel("Time [{}]".format(frames.frame_report.time_units))
        ax.set_xlim([t_start, t_stop])

    ids = frames.nodes.ids(group)
    if plot_type == "mean":
        data = frames.get(group=ids, t_start=t_start, t_stop=t_stop).T
        ax.plot(data.mean())
    elif plot_type == "all":
        if len(ids) > 15:
            L.warning('PopulationSomaReport.trace: Sample too big. Keep the first 15 node_ids.')
        data = frames.get(group=ids[:15], t_start=t_start, t_stop=t_stop).T
        for _, row in data.iterrows():
            ax.plot(row)
    return ax
