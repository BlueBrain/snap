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
from more_itertools import roundrobin

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


def spikes_firing_rate_histogram(filtered_report, time_binsize=None, ax=None):  # pragma: no cover
    """Spike firing rate histogram.

    This plot shows the number of nodes firing during a range of time.

    Args:
        time_binsize(None/int/float): bin size (milliseconds). If None, a binning heuristic is used
            to create an histogram with ~100 spikes per bin in average.
        ax(matplotlib.Axis): matplotlib Axis to draw on (if not specified, pyplot.gca() is used).

    Returns:
        matplotlib.Axis: Axis containing firing rate histogram.

    Notes:
        If no axis is provided through the ax=ax keyword argument,
        then a default layout is set using pyplot.gca().
    """
    # pylint: disable=too-many-locals
    plt = _get_pyplot()
    if time_binsize is not None and time_binsize <= 0:
        raise BluepySnapError("Invalid time_binsize = {}. Should be > 0.".format(time_binsize))

    spike_report = filtered_report.spike_report

    times = filtered_report.report.index
    node_count = filtered_report.report[['ids', 'population']].drop_duplicates().shape[0]

    if len(times) == 0:
        raise BluepySnapError("No data to display. You should check your "
                              "'group' query: {}.".format(spike_report.group))

    time_start = np.min(times)
    time_stop = np.max(times)

    if time_binsize is None:
        # heuristic for a nice bin size (~100 spikes per bin on average)
        time_binsize = min(50.0, (time_stop - time_start) / ((len(times) / 100.) + 1.))

    bins = np.append(np.arange(time_start, time_stop, time_binsize), time_stop)
    hist, bin_edges = np.histogram(times, bins=bins)
    freq = 1.0 * hist / node_count / (0.001 * time_binsize)

    if ax is None:
        ax = plt.gca()
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('PSTH [Hz]')

    # use the middle of the bins instead of the start of the bin
    ax.plot(0.5 * (bin_edges[1:] + bin_edges[:-1]), freq, label="PSTH", drawstyle='steps-mid')
    return ax


def spike_raster(filtered_report, y_axis=None, ax=None):  # pragma: no cover
    """Spike raster plot.

    Shows a global overview of the circuit's firing nodes. The y axis can project either the
    node_ids or any properties present in the different node populations.

    Args:
        y_axis (None/str): The property to display on the y axis. None is node_ids.
        ax(matplotlib.Axis): matplotlib Axis to draw on (if not specified, pyplot.gca() is used).

    Returns:
        matplotlib.Axis: Axis containing Spikes raster plot.

    Notes:
        If no axis is provided through the ax=ax keyword argument,
        then a default layout is set using pyplot.gca().
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    plt = _get_pyplot()

    spike_report = filtered_report.spike_report
    population_names = filtered_report.spike_report.population_names

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
        elif pd.api.types.is_categorical_dtype(spikes.nodes.property_dtypes[y_axis]):
            props["categorical_values"].update(spikes.nodes.property_values(y_axis))
        else:
            props["ymin"] = min(props["ymin"], spikes.nodes.get(properties=y_axis).min())
            props["ymax"] = max(props["ymax"], spikes.nodes.get(properties=y_axis).max())

    report = filtered_report.report

    dtype = spike_report[population_names[0]].nodes.property_dtypes[y_axis] if y_axis else None
    if dtype and pd.api.types.is_categorical_dtype(dtype):
        # this is to prevent the problems when concatenating categoricals with unknown categories
        dtype = str
    data = pd.Series(index=report.index, dtype=dtype)
    for population in population_names:
        spikes = spike_report[population]
        mask = report["population"] == population
        if y_axis is None:
            data.loc[mask] = report.loc[mask, "ids"] + props["node_id_offset"]
        else:
            ids = report.loc[mask, "ids"].to_numpy()
            try:
                ys = spikes.nodes.get(properties=y_axis)
            except BluepySnapError:
                continue
            # astype is used to avoid problems with the categorical
            data[mask] = ys[ids].astype(dtype).to_numpy()
        _update_raster_properties()

    data = data[data.notna()]
    if ax is None:
        ax = plt.gca()
        ax.xaxis.grid()
        ax.set_xlabel("Time [ms]")
        ax.tick_params(axis='y', which='both', length=0)
        ax.set_xlim(spike_report.time_start, spike_report.time_stop)
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


def spikes_isi(filtered_report, use_frequency=False, binsize=None, ax=None):  # pragma: no cover
    # pylint: disable=too-many-locals
    """Interspike interval histogram.

    This plots show the binned time/frequency interval between to spikes for neurons.

    Args:
        use_frequency(bool): use inverse interspike interval times (Hz)
        binsize(None/int/float): bin size in milliseconds or Hz. If None is used the binning is
            delegated to matplolib and is done automatically.
        ax(matplotlib.Axis): matplotlib Axis to draw on (if not specified, pyplot.gca() is used).

    Returns:
        matplotlib.Axis: axis containing the interspike interval histogram.

    Notes:
        If no axis is provided through the ax=ax keyword argument,
        then a default layout is set using pyplot.gca().
    """
    plt = _get_pyplot()
    if binsize is not None and binsize <= 0:
        raise BluepySnapError("Invalid binsize = {}. Should be > 0.".format(binsize))

    gb = filtered_report.report.groupby(["ids", "population"])
    values = np.concatenate([np.diff(node_spikes.index.to_numpy()) for _, node_spikes in gb])

    if len(values) == 0:
        raise BluepySnapError("No data to display. You should check your "
                              "'group' query: {}.".format(filtered_report.spike_report.group))
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


def spikes_firing_animation(filtered_report, x_axis=Node.X, y_axis=Node.Y,
                            dt=20, ax=None):  # pragma: no cover
    # pylint: disable=too-many-locals,too-many-arguments,anomalous-backslash-in-string
    """Simple animation of simulation spikes.

    Each frame of the animation represents the spiking nodes during a period of dt ms seconds
    in a coordinate system corresponding to the x, y or z axis of the circuit.

    Args:
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
        """Verifies axes values."""
        axes = {Node.X, Node.Y, Node.Z}
        if axis not in axes:
            raise BluepySnapError('{} is not a valid axis'.format(axis))

    _check_axis(x_axis)
    _check_axis(y_axis)

    spike_report = filtered_report.spike_report
    population_names = filtered_report.spike_report.population_names
    report = filtered_report.report

    data = pd.DataFrame(index=report.index, columns=[x_axis, y_axis], dtype=np.float32)

    for population in population_names:
        spikes = spike_report[population]
        pop_mask = report["population"] == population

        ids = report.loc[pop_mask, "ids"].to_numpy()
        try:
            values = spikes.nodes.get(properties=[x_axis, y_axis]).loc[ids].to_numpy()
            data.loc[pop_mask, [x_axis, y_axis]] = values
        except BluepySnapError:
            continue

    data = data[data.notna()]

    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
        ax.set_title('time = {}ms'.format(np.min(data.index)))
        x_limits = [data[x_axis].min(), data[x_axis].max()]
        y_limits = [data[y_axis].min(), data[y_axis].max()]
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_xlabel(r'{} $\mu$m'.format(x_axis))  # noqa
        ax.set_ylabel(r'{} $\mu$m'.format(y_axis))  # noqa

    else:
        fig = ax.figure

    dots = ax.plot([], [], '.k')

    def update_animation(frame):
        """Update the animation plots and axes."""
        ax.set_title('time = ' + str(frame * dt) + ' ms')
        mask = (data.index >= frame * dt) & (data.index <= (frame + 1) * dt)
        positions = data.loc[mask, [x_axis, y_axis]].values
        x = positions[:, 0]
        y = positions[:, 1]
        dots[0].set_data(x, y)
        return dots

    frames = list(range(int(data.index[0] / dt), int(data.index[-1] / dt)))
    anim = FuncAnimation(fig, update_animation, frames=frames)
    return anim, ax


def frame_trace(filtered_report, plot_type='mean', ax=None):  # pragma: no cover
    """Returns a plot displaying the voltage of a node or a compartment as a function of time.

    Args:
        plot_type (str): string either `all` or `mean`. `all` will plot the first 15 traces from the
            group. `mean` will plot the mean value of the node
        ax: A plot axis object that will be updated

    Returns:
        matplotlib.Axis: axis containing the soma's traces.
    """
    # pylint: disable=too-many-locals

    plt = _get_pyplot()

    if ax is None:
        ax = plt.gca()
        data_units = filtered_report.frame_report.data_units
        if plot_type == "mean":
            ax.set_ylabel('Avg volt. [{}]'.format(data_units))
        elif plot_type == "all":
            ax.set_ylabel('Voltage [{}]'.format(data_units))
        ax.set_xlabel("Time [{}]".format(filtered_report.frame_report.time_units))
        ax.set_xlim([filtered_report.report.index.min(), filtered_report.report.index.max()])

    if plot_type == "mean":
        ax.plot(filtered_report.report.T.mean())
    elif plot_type == "all":
        max_per_pop = 15
        levels = filtered_report.report.columns.levels
        slicer = []
        # create a slicer that will slice only on the last level of the columns
        # that is, node_id for the soma report, element_id for the compartment report
        for i, _ in enumerate(levels):
            max_ = levels[i][:max_per_pop][-1]
            slicer.append(slice(None) if i != len(levels) - 1 else slice(None, max_))
        data = filtered_report.report.loc[:, tuple(slicer)].T
        # create [[(pop1, id1), (pop1, id2),...], [(pop2, id1), (pop2, id2),...]]
        indexes = [[(pop, idx) for idx in data.loc[pop].index] for pop in levels[0]]
        # try to keep the maximum of ids from each population
        kept_ids = list(roundrobin(*indexes))[:max_per_pop]
        for _, row in data.loc[kept_ids].iterrows():
            ax.plot(row)
    else:
        raise BluepySnapError("Unknown plot_type {}. Should be 'mean or 'all'.".format(plot_type))
    return ax
