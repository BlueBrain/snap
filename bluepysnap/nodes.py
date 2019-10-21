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

"""Node population access."""

import collections

import libsonata
import numpy as np
import pandas as pd
import six

from cached_property import cached_property

from bluepysnap import utils
from bluepysnap.exceptions import BlueSnapError

DYNAMICS_PREFIX = "@dynamics:"


def _get_population_name(h5_filepath):
    populations = libsonata.NodeStorage(h5_filepath).population_names
    if len(populations) != 1:
        raise BlueSnapError(
            "Only single-population node collections are supported (found: %d)" % len(populations)
        )
    return list(populations)[0]


def _load_population(h5_filepath, csv_filepath, population):
    """Load node properties from SONATA Nodes.

    Returns:
        pandas.DataFrame with node properties (zero-based index).
    """
    nodes = libsonata.NodePopulation(h5_filepath, csv_filepath or '', population)

    node_count = nodes.size
    result = pd.DataFrame(index=np.arange(node_count))

    _all = libsonata.Selection([(0, node_count)])
    for attr in sorted(nodes.attribute_names):
        result[attr] = nodes.get_attribute(attr, _all)
    for attr in sorted(nodes.dynamics_attribute_names):
        result['%s%s' % (DYNAMICS_PREFIX, attr)] = nodes.get_dynamics_attribute(attr, _all)

    return result


# TODO: move to `libsonata` library
def _complex_query(prop, query):
    # pylint: disable=assignment-from-no-return
    result = np.full(len(prop), True)
    for key, value in six.iteritems(query):
        if key == '$regex':
            result = np.logical_and(result, prop.str.match(value + "\\Z"))
        else:
            raise BlueSnapError("Unknown query modifier: '%s'" % key)
    return result


def _node_ids_by_filter(nodes, props):
    """Return index of `nodes` rows matching `props` dict.

    `props` values could be:
        pairs (range match for floating dtype fields)
        scalar or iterables (exact or "one of" match for other fields)

    E.g.:
        >>> _node_ids_by_filter(nodes, { Node.X: (0, 1), Node.MTYPE: 'L1_SLAC' })
        >>> _node_ids_by_filter(nodes, { Node.LAYER: [2, 3] })
    """
    # pylint: disable=assignment-from-no-return
    unknown_props = set(props) - set(nodes.columns)
    if unknown_props:
        raise BlueSnapError("Unknown node properties: [{0}]".format(", ".join(unknown_props)))

    mask = np.full(len(nodes), True)
    for prop, values in six.iteritems(props):
        prop = nodes[prop]
        if issubclass(prop.dtype.type, np.floating):
            v1, v2 = values
            prop_mask = np.logical_and(prop >= v1, prop <= v2)
        elif isinstance(values, six.string_types) and values.startswith('regex:'):
            prop_mask = _complex_query(prop, {'$regex': values[6:]})
        elif isinstance(values, collections.Mapping):
            prop_mask = _complex_query(prop, values)
        else:
            prop_mask = np.in1d(prop, values)
        mask = np.logical_and(mask, prop_mask)

    return nodes.index[mask].values


class NodePopulation(object):
    """Node population access."""

    def __init__(self, config, circuit):
        """Initializes a NodePopulation object from a config dictionary and a circuit."""
        self._h5_filepath = config['nodes_file']
        self._csv_filepath = config['node_types_file']
        if 'node_sets_file' in config:
            self._node_sets = utils.load_json(config['node_sets_file'])
        else:
            self._node_sets = {}
        self._circuit = circuit

    @cached_property
    def name(self):
        """Node population name."""
        return _get_population_name(self._h5_filepath)

    @property
    def size(self):
        """Node population size."""
        return len(self._data)

    @property
    def property_names(self):
        """Set of available node properties."""
        return set(self._data.columns)

    def property_values(self, prop):
        """Set of values for a given property.

        Args:
           prop (str): Name of the property to retrieve.

        Returns:
            set: A set of the unique values of the property in the node population.
        """
        return set(self.get(properties=prop).unique())

    @property
    def node_sets(self):
        """Node sets defined for this node population."""
        return self._node_sets

    @cached_property
    def _data(self):
        return _load_population(self._h5_filepath, self._csv_filepath, self.name)

    def _check_id(self, node_id):
        """Check that single node ID belongs to the circuit."""
        if node_id not in self._data.index:
            raise BlueSnapError("node ID not found: %d" % node_id)

    def _check_ids(self, node_ids):
        """Check that node IDs belong to the circuit."""
        missing = pd.Index(node_ids).difference(self._data.index)
        if not missing.empty:
            raise BlueSnapError("node ID not found: [%s]" % ",".join(map(str, missing)))

    def _check_property(self, prop):
        if prop not in self.property_names:
            raise BlueSnapError("No such property: '%s'" % prop)

    def ids(self, group=None, limit=None, sample=None):
        """Node IDs corresponding to node ``group``.

        Args:
            group (int/sequence/str/mapping/None): Which IDs will be returned
                depends on the type of the ``group`` argument:

                - ``int``: return a single node ID if it belongs to the circuit.
                - ``sequence``: returns a list of node IDs.
                - ``str``: returns a target name.
                - ``mapping``: returns IDs of nodes matching a properties filter.
                - ``None``: returns all node IDs.

                If ``group`` is a ``sequence``, the order of results is preserved.
                Otherwise the result is sorted and contains no duplicates.

            sample (int): If specified, randomly choose ``sample`` number of
                IDs from the match result.

            limit (int): If specified, return the first ``limit`` number of
                IDs from the match result.

        Returns:
            numpy.array: A numpy array of IDs.
        """
        preserve_order = False

        if isinstance(group, six.string_types):
            if group not in self._node_sets:
                raise BlueSnapError("Undefined node set: %s" % group)
            group = self._node_sets[group]

        if group is None:
            result = self._data.index.values
        elif isinstance(group, collections.Mapping):
            result = _node_ids_by_filter(self._data, props=group)
        elif isinstance(group, np.ndarray):
            result = group
            self._check_ids(result)
            preserve_order = True
        else:
            result = utils.ensure_list(group)
            self._check_ids(result)
            preserve_order = isinstance(group, collections.Sequence)

        if sample is not None:
            result = np.random.choice(result, sample, replace=False)
            preserve_order = False
        if limit is not None:
            result = result[:limit]

        result = np.array(result, dtype=int)
        if preserve_order:
            return result
        else:
            return np.unique(result)

    def get(self, group=None, properties=None):
        """Node properties as a pandas Series or DataFrame.

        Args:
            group (int/sequence/str/mapping/None): Which nodes will have their properties
                returned depends on the type of the ``group`` argument:

                - ``int``: return the properties of a single node.
                - ``sequence``: return the properties from a list of node.
                - ``str``: return the properties of a target name.
                - ``mapping``: return the properties of nodes matching a properties filter.
                - ``None``: return the properties of all nodes.

            properties (set): If specified, return only the properties in the set.
                Otherwise return all properties.

        Returns:
            pandas.Series/pandas.DataFrame:
                If single node ID is passed as ``group`` returns a pandas Series.
                Otherwise return a pandas DataFrame indexed by node IDs.
        """
        result = self._data

        if properties is not None:
            for p in utils.ensure_list(properties):
                self._check_property(p)
            result = result[properties]

        if group is not None:
            if isinstance(group, six.integer_types + (np.integer,)):
                self._check_id(group)
            else:
                group = self.ids(group)
            result = result.loc[group]

        return result

    def positions(self, group=None):
        """Node position(s) as pandas Series or DataFrame.

        Args:
            group (int/sequence/str/mapping/None): Which nodes will have their positions
                returned depends on the type of the ``group`` argument:

                - ``int``: return the position of a single node.
                - ``sequence``: return the positions from a list of node IDs.
                - ``str``: return the positions of a target name.
                - ``mapping``: return the positions of nodes matching a properties filter.
                - ``None``: return the positions of all nodes.

        Returns:
            pandas.Series/pandas.DataFrame:
                Series of ('x', 'y', 'z') if single node ID is

                passed as ``group``. Otherwise, a pandas.DataFrame of ('x', 'y', 'z') indexed
                by node IDs.
        """
        result = self.get(group=group, properties=['x', 'y', 'z'])
        return result.astype(float)

    def orientations(self, group=None):
        """Node orientation(s) as a pandas Series or DataFrame.

        Args:
            group (int/sequence/str/mapping/None): Which nodes will have their positions
                returned depends on the type of the ``group`` argument:

                - ``int``: return the orientation of a single node.
                - ``sequence``: return the orientations from a list of node IDs.
                - ``str``: return the orientations of a target name.
                - ``mapping``: return the orientations of nodes matching a properties filter.
                - ``None``: return the orientations of all nodes.

        Returns:
            numpy.ndarry/pandas.Series:
                A 3x3 rotation matrix if a single node ID is passed as ``group``.
                Otherwise a pandas Series with rotation matrices indexed by node IDs.
        """
        props = [
            'rotation_angle_xaxis',
            'rotation_angle_yaxis',
            'rotation_angle_zaxis',
        ]
        result = self.get(group=group, properties=props)
        if isinstance(group, six.integer_types + (np.integer,)):
            result = utils.euler2mat(
                [result['rotation_angle_zaxis']],
                [result['rotation_angle_yaxis']],
                [result['rotation_angle_xaxis']],
            )[0]
        else:
            result = pd.Series(
                utils.euler2mat(
                    result['rotation_angle_zaxis'].values,
                    result['rotation_angle_yaxis'].values,
                    result['rotation_angle_xaxis'].values,
                ),
                index=result.index,
                name='orientation'
            )
        return result

    def count(self, group=None):
        """Total number of nodes for a given node group.

        Args:
            group (int/sequence/str/mapping/None): Which nodes will have their positions
                returned depends on the type of the ``group`` argument:

                - ``int``: return the count of a single node.
                - ``sequence``: return the count of nodes from a list of node IDs.
                - ``str``: return the count of nodes of a target name.
                - ``mapping``: return count of nodes matching a properties filter.
                - ``None``: return the count of all nodes.

        Returns:
            int: The total number of nodes in a given group.
        """
        return len(self.ids(group))

    @cached_property
    def morph(self):
        """Access to node morphologies."""
        from bluepysnap.morph import MorphHelper
        return MorphHelper(
            self._circuit.config['components']['morphologies_dir'],
            self
        )
