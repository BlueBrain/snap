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
import pandas as pd

from bluepysnap.exceptions import BluepySnapError
from bluepysnap.sonata_constants import Node
from bluepysnap.utils import roundrobin


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


def _get_spike_report(spike_report):
    """Take a PopulationSpikeReport or a SpikeReport and return a SpikeReport with population names.

    Args:
        spike_report (PopulationSpikeReport/SpikeReport): the simulation spike object to transform.

    Returns:
        tuple: a SpikeReport and a sorted list of the populations

    Notes:
        Avoids circular imports (no import of PopulationSpikeReport/SpikeReport inside the
        _plotting module), allows simple factorizations for plot functions and provides a
        sorted list of population names.
    """
    try:
        obj = spike_report.spike_report
        population_names = [spike_report.name]
    except AttributeError:
        obj = spike_report
        population_names = sorted(obj.population_names)
    return obj, population_names


def spikes_firing_rate_histogram(spike_obj, group=None, t_start=None, t_stop=None,
                                 binsize=None, ax=None):  # pragma: no cover
    """PopulationSpikeReport/SpikeReport firing rate histogram.

    Args:
        group (None/int/list/np.array/dict): Get spikes filtered by group. See NodePopulation.ids().
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
    # pylint: disable=too-many-locals
    plt = _get_pyplot()
    obj, population_names = _get_spike_report(spike_obj)

    times = np.empty((0,))
    node_count = 0
    for population in population_names:
        spikes = obj[population]
        try:
            ids = spikes.nodes.ids(group=group)
        except BluepySnapError:
            continue
        data = spikes.get(group=ids, t_start=t_start, t_stop=t_stop)
        if isinstance(data, np.ndarray):
            node_count += 1
        else:
            node_count += data.nunique()
            data = data.index.to_numpy()
        times = np.concatenate([times, data])

    if len(times) == 0:
        raise BluepySnapError("No data to display. You should check your "
                              "'group' query: {}.".format(group))

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


def spike_raster(spike_obj, group=None, t_start=None, t_stop=None, y_axis=None,
                 ax=None):  # pragma: no cover
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    """PopulationSpikeReport/SpikeReport raster plot.

    Args:
        group (None/int/list/np.array/dict): Get spikes filtered by group. See NodePopulation.ids().
        y_axis (None/str): The property to display on the y axis. None is node_ids.
        t_start (None/float): Include only spikes occurring after this time.
        t_stop (None/float): Include only spikes occurring before this time.
        ax(matplotlib.Axis): matplotlib Axis to draw on (if not specified, pyplot.gca() is used).

    Returns:
        matplotlib.Axis: Axis containing Spikes raster plot.

    Notes:
        If no axis is provided through the ax=ax keyword argument,
        then a default layout is set using pyplot.gca().
    """
    plt = _get_pyplot()

    obj, population_names = _get_spike_report(spike_obj)

    props = {"node_id_offset": 0,
             "pop_separators": [],
             "categorical_values": set(),
             "ymin": np.inf,
             "ymax": -np.inf
             }

    def _update_raster_properties():
        if y_axis is None:
            props["node_id_offset"] += spikes.nodes.size
            props["pop_separators"].append(props["node_id_offset"])
        elif np.issubdtype(spikes.nodes.property_dtypes[y_axis], np.number):
            props["ymin"] = min(props["ymin"], spikes.nodes.get(properties=y_axis).min())
            props["ymax"] = max(props["ymax"], spikes.nodes.get(properties=y_axis).max())
        else:
            props["categorical_values"].update(spikes.nodes.property_values(y_axis))

    data = pd.Series()
    for population in population_names:
        spikes = obj[population]
        if y_axis is not None and y_axis not in spikes.nodes.property_names:
            raise BluepySnapError("Cannot plot raster with y_axis={}. Parameter not present in the "
                                  "{} population".format(y_axis, population))
        try:
            # try to resolve the group (some field in group may be missing for population)
            ids = spikes.nodes.ids(group=group)
        except BluepySnapError:
            # need to keep all the values from the population even if cannot resolve group
            _update_raster_properties()
            continue
        population_data = spikes.get(group=ids, t_start=t_start, t_stop=t_stop)
        ts = population_data.index
        if y_axis is None:
            ys = population_data.to_numpy() + props["node_id_offset"]
        else:
            ys = spikes.nodes.get(properties=y_axis).loc[population_data.to_numpy()].to_numpy()
        _update_raster_properties()
        population_data = pd.Series(data=ys, index=ts)
        data = pd.concat([population_data, data])

    data.sort_index(inplace=True)
    if len(data) == 0:
        raise BluepySnapError("No data to display. You should check your "
                              "'group' query: {}.".format(group))

    if ax is None:
        ax = plt.gca()
        ax.xaxis.grid()
        ax.set_xlabel("Time [ms]")
        ax.tick_params(axis='y', which='both', length=0)
        ax.set_xlim(obj.time_start, obj.time_stop)
        if y_axis is None:
            ax.set_ylim(0, props["node_id_offset"])
            ax.set_ylabel("nodes")
        else:
            if np.issubdtype(type(data.iloc[0]), np.number):
                # automatically expended by plt if ymin == ymax
                ax.set_ylim(props["ymin"], props["ymax"])
            else:
                labels = sorted(list(props["categorical_values"]))
                ax.set_yticks(np.arange(len(labels)))
                ax.set_yticklabels(labels)
                if len(labels) > 1:
                    ax.set_ylim(-0.5, len(labels) - 0.5)
            ax.set_ylabel("{}".format(y_axis))

    ax.scatter(data.index.to_numpy(), data.to_numpy(), s=10, marker='|')
    if len(props["pop_separators"]) > 1:
        for separator in props["pop_separators"]:
            ax.axhline(y=separator, color='black', lw=2)
    return ax


def spikes_isi(spike_obj, group=None, t_start=None, t_stop=None,
               use_frequency=False, binsize=None, ax=None):  # pragma: no cover
    # pylint: disable=too-many-locals
    """PopulationSpikeReport/SpikeReport  Interspike interval histogram.

    Args:
        group (None/int/list/np.array/dict): Get spikes filtered by group. See NodePopulation.ids().
        t_start (float): Include only spikes occurring after this time.
        t_stop (float): Include only spikes occurring before this time.
        use_frequency(bool): use inverse interspike interval times (Hz)
        binsize(float): bin size (milliseconds or Hz)
        ax(matplotlib.Axis): matplotlib Axis to draw on (if not specified, pyplot.gca() is used).

    Returns:
        matplotlib.Axis: axis containing the interspike interval histogram.

    Notes:
        If no axis is provided through the ax=ax keyword argument,
        then a default layout is set using pyplot.gca().
    """
    plt = _get_pyplot()
    obj, population_names = _get_spike_report(spike_obj)
    values = np.empty((0,))
    for population in population_names:
        spikes = obj[population]
        try:
            ids = spikes.nodes.ids(group=group)
        except BluepySnapError:
            continue

        data = spikes.get(group=ids, t_start=t_start, t_stop=t_stop)
        if len(data) == 0:
            continue
        data = np.concatenate([np.diff(node_spikes.index.to_numpy())
                               for _, node_spikes in data.groupby(data)])
        values = np.concatenate([values, data])

    if len(values) == 0:
        raise BluepySnapError("No data to display. You should check your "
                              "'group' query: {}.".format(group))
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


def spikes_firing_animation(spike_obj, group=None, t_start=None,
                            t_stop=None, x_axis=Node.X, y_axis=Node.Y,
                            dt=20, ax=None):  # pragma: no cover
    # pylint: disable=too-many-locals,too-many-arguments,anomalous-backslash-in-string
    """PopulationSpikeReport/SpikeReport  simple animation of simulation spikes.

    Args:
        group (None/int/list/np.array/dict): Get spikes filtered by group. See NodePopulation.ids().
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
    obj, population_names = _get_spike_report(spike_obj)

    def _check_axis(axis):
        """Verifies axes values."""
        axes = {Node.X, Node.Y, Node.Z}
        if axis not in axes:
            raise BluepySnapError('{} is not a valid axis'.format(axis))

    _check_axis(x_axis)
    _check_axis(y_axis)

    offset = 0
    positions = pd.DataFrame()
    data = pd.Series()
    for population in population_names:
        spikes = obj[population]
        try:
            # time and population node_ids
            population_data = spikes.get(group=group, t_start=t_start, t_stop=t_stop)
            active_gids = population_data.unique()
            # positions indexed by population node_ids
            population_positions = spikes.nodes.get(group=active_gids, properties=[x_axis, y_axis])
            # reindex the node_ids index with the offset
            population_positions.index += offset
            # non overlapping indices for positions
            positions = pd.concat([positions, population_positions])
            # non overlapping node_ids for the times
            data = pd.concat([data, population_data + offset])
            offset += spikes.nodes.size
        except BluepySnapError:
            offset += spikes.nodes.size
            continue

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
        """Update the animation plots and axes."""
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


def _get_frame_report(frame_report):
    """Take a PopulationFrameReport or a FrameReport and return a FrameReport with population names.

    Args:
        frame_report (PopulationFrameReport/FrameReport): the simulation frame object to transform.

    Returns:
        tuple: a FrameReport and a sorted list of the populations

    Notes:
        Avoids circular imports (no import of PopulationFrameReport/FrameReport inside the
        _plotting module), allows simple factorizations for plot functions and provides a
        sorted list of population names.
    """
    try:
        obj = frame_report.frame_report
        population_names = [frame_report.name]
    except AttributeError:
        obj = frame_report
        population_names = sorted(obj.population_names)
    return obj, population_names


def soma_trace(frame_report, group=None, t_start=None, t_stop=None, plot_type='mean',
               ax=None):  # pragma: no cover
    # pylint: disable=too-many-locals
    """Return PopulationSomaReport potential plot displaying the voltage as a function of time.

    Args:
        group (None/int/list/np.array/dict): Get spikes filtered by group. See NodePopulation.ids().
        t_start (float): Include only spikes occurring after this time.
        t_stop (float): Include only spikes occurring before this time.
        plot_type (str): string either `all` or `mean`. `all` will plot the first 15 traces from the
        group. `mean` will plot the mean value of the node
        ax: A plot axis object that will be updated

    Returns:
        matplotlib.Axis: axis containing the soma's traces.
    """
    plt = _get_pyplot()
    obj, population_names = _get_frame_report(frame_report)

    max_per_pop = 15 if plot_type != "mean" else -1
    data = pd.DataFrame()
    offset = 0
    population_ids = []
    for population in population_names:
        frames = obj[population]
        try:
            ids = frames.nodes.ids(group)
        except BluepySnapError:
            offset += frames.nodes.size
            continue

        population_data = frames.get(group=ids[:max_per_pop], t_start=t_start, t_stop=t_stop).T
        population_data.index += offset
        if plot_type != "mean":
            population_ids.append(population_data.index)
        offset += frames.nodes.size
        data = pd.concat([population_data, data])

    if ax is None:
        ax = plt.gca()
        data_units = obj.data_units
        if plot_type == "mean":
            ax.set_ylabel('Avg volt. [{}]'.format(data_units))
        elif plot_type == "all":
            ax.set_ylabel('Voltage [{}]'.format(data_units))
        ax.set_xlabel("Time [{}]".format(obj.time_units))
        ax.set_xlim([t_start, t_stop])

    if plot_type == "mean":
        ax.plot(data.mean())
    else:
        # try to keep the maximum of ids from each population
        kept_ids = list(roundrobin(*population_ids))[:max_per_pop]
        for _, row in data.loc[kept_ids].iterrows():
            ax.plot(row)
    return ax
