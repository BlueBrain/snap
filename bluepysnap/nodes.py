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
import inspect

import libsonata
import numpy as np
import pandas as pd
import six

from cached_property import cached_property

from bluepysnap import utils
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.sonata_constants import DYNAMICS_PREFIX, NODE_ID_KEY, Node, ConstContainer


class NodeStorage(object):
    """Node storage access."""
    def __init__(self, config, circuit):
        """Initializes a NodeStorage object from a node config and a Circuit.

        Args:
            config (dict): a node config from the global circuit config
            circuit (bluepysnap.Circuit): the circuit object that contains the NodePopulation
            from this storage.

        Returns:
            NodeStorage: A NodeStorage object.
        """
        self._h5_filepath = config['nodes_file']
        self._csv_filepath = config['node_types_file']
        if 'node_sets_file' in config:
            self._node_sets = utils.load_json(config['node_sets_file'])
        else:
            self._node_sets = {}
        self._circuit = circuit
        self._populations = {}

    @cached_property
    def storage(self):
        """Access to the libsonata node storage."""
        return libsonata.NodeStorage(self._h5_filepath)

    @cached_property
    def population_names(self):
        """Returns all population names inside this file."""
        return self.storage.population_names

    @property
    def circuit(self):
        """Returns the circuit object containing this storage."""
        return self._circuit

    @property
    def node_sets(self):
        """Returns the node sets defined for this node population."""
        return self._node_sets

    def population(self, population_name):
        """Access the different populations from the storage."""
        if population_name not in self._populations:
            self._populations[population_name] = NodePopulation(self, population_name)
        return self._populations[population_name]

    def load_population_data(self, population):
        """Load node properties from SONATA Nodes.

        Args:
            population (str): a population name .

        Returns:
            pandas.DataFrame with node properties (zero-based index).
        """
        nodes = self.storage.open_population(population)

        node_count = nodes.size
        result = pd.DataFrame(index=np.arange(node_count))

        _all = libsonata.Selection([(0, node_count)])
        for attr in sorted(nodes.attribute_names):
            result[attr] = nodes.get_attribute(attr, _all)
        for attr in sorted(utils.add_dynamic_prefix(nodes.dynamics_attribute_names)):
            result[attr] = nodes.get_dynamics_attribute(attr.split(DYNAMICS_PREFIX)[1], _all)
        return result


# TODO: move to `libsonata` library
def _complex_query(prop, query):
    # pylint: disable=assignment-from-no-return
    result = np.full(len(prop), True)
    for key, value in six.iteritems(query):
        if key == '$regex':
            result = np.logical_and(result, prop.str.match(value + "\\Z"))
        else:
            raise BluepySnapError("Unknown query modifier: '%s'" % key)
    return result


def _node_ids_by_filter(node_data, props):
    """Return index of `nodes` rows matching `props` dict.

    `props` values could be:
        pairs (range match for floating dtype fields)
        scalar or iterables (exact or "one of" match for other fields)

    E.g.:
        >>> _node_ids_by_filter(node_data, { Node.X: (0, 1), Node.MTYPE: 'L1_SLAC' })
        >>> _node_ids_by_filter(node_data, { Node.LAYER: [2, 3] })
    """
    # pylint: disable=assignment-from-no-return
    unknown_props = set(props) - set(node_data.columns)
    if unknown_props:
        raise BluepySnapError("Unknown node properties: [{0}]".format(", ".join(unknown_props)))

    mask = np.full(len(node_data), True)
    for prop, values in six.iteritems(props):
        prop = node_data[prop]
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

    return node_data.index[mask].values


class NodePopulation(object):
    """Node population access."""

    def __init__(self, node_storage, population_name):
        """Initializes a NodePopulation object from a NodeStorage and population name.

        Args:
            node_storage (NodeStorage): the node storage containing the node population
            population_name (str): the name of the node population
        Returns:
            NodePopulation: A NodePopulation object.
        """
        self._node_storage = node_storage
        self.name = population_name

    @property
    def node_sets(self):
        """Node sets defined for this node population."""
        return self._node_storage.node_sets

    @cached_property
    def _data(self):
        """Collected data for the node population as a pandas.DataFrame."""
        return self._node_storage.load_population_data(self.name)

    @cached_property
    def _population(self):
        return self._node_storage.storage.open_population(self.name)

    @cached_property
    def size(self):
        """Node population size."""
        return self._population.size

    @cached_property
    def _property_names(self):
        return set(self._population.attribute_names)

    @cached_property
    def _dynamics_params_names(self):
        return set(utils.add_dynamic_prefix(self._population.dynamics_attribute_names))

    @property
    def property_names(self):
        """Set of available node properties."""
        return self._property_names | self._dynamics_params_names

    def container_property_names(self, container):
        """Lists the ConstContainer properties shared with the NodePopulation.

        Args:
            container (ConstContainer): a container class for node properties.

        Returns:
            list: A list of strings corresponding to the properties that you can use from the
                container class

        Examples:
            >>> from bluepysnap.sonata_constants import Node
            >>> print(my_node_population.container_property_names(Node))
            >>> ["X", "Y", "Z"] # values from Node that you can use with my_node_population
            >>> my_node_population.property_values(Node.X)
            >>> my_node_population.property_values(Node.get("X"))
        """
        if not inspect.isclass(container) or not issubclass(container, ConstContainer):
            raise BluepySnapError("'container' must be a subclass of ConstContainer")
        in_file = self.property_names
        return [k for k in container.key_set() if container.get(k) in in_file]

    def property_values(self, prop):
        """Set of values for a given property.

        Args:
           prop (str): Name of the property to retrieve.

        Returns:
            set: A set of the unique values of the property in the node population.
        """
        return set(self.get(properties=prop).unique())

    def _check_id(self, node_id):
        """Check that single node ID belongs to the circuit."""
        if node_id not in self._data.index:
            raise BluepySnapError("node ID not found: %d" % node_id)

    def _check_ids(self, node_ids):
        """Check that node IDs belong to the circuit."""
        missing = pd.Index(node_ids).difference(self._data.index)
        if not missing.empty:
            raise BluepySnapError("node ID not found: [%s]" % ",".join(map(str, missing)))

    def _check_property(self, prop):
        if prop not in self.property_names:
            raise BluepySnapError("No such property: '%s'" % prop)

    def ids(self, group=None, limit=None, sample=None):
        """Node IDs corresponding to node ``group``.

        Args:
            group (int/sequence/str/mapping/None): Which IDs will be returned
                depends on the type of the ``group`` argument:

                - ``int``: return a single node ID if it belongs to the circuit.
                - ``sequence``: return IDs of nodes in an array.
                - ``str``: return IDs of nodes in a node set.
                - ``mapping``: return IDs of nodes matching a properties filter.
                - ``None``: return all node IDs.

                If ``group`` is a ``sequence``, the order of results is preserved.
                Otherwise the result is sorted and contains no duplicates.

            sample (int): If specified, randomly choose ``sample`` number of
                IDs from the match result.

            limit (int): If specified, return the first ``limit`` number of
                IDs from the match result.

        Returns:
            numpy.array: A numpy array of IDs.
        """
        # pylint: disable=too-many-branches
        preserve_order = False
        node_filter = slice(None, None, 1)
        if isinstance(group, six.string_types):
            if group not in self.node_sets:
                raise BluepySnapError("Undefined node set: %s" % group)
            group = self.node_sets[group]
            if not isinstance(group, collections.MutableMapping):
                raise BluepySnapError("Node set values must be dict not: %s" % type(group))
            if len(group) == 0:
                group = None
            elif NODE_ID_KEY in group:
                node_filter = group.pop(NODE_ID_KEY)
                node_filter = utils.ensure_list(node_filter)
                if not group:
                    group = np.asarray(node_filter)

        if group is None:
            result = self._data.index.values
        elif isinstance(group, collections.Mapping):
            result = _node_ids_by_filter(self._data.loc[node_filter], props=group)
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
                - ``str``: return the properties of nodes in a node set.
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
                - ``str``: return the positions of nodes in a node set.
                - ``mapping``: return the positions of nodes matching a properties filter.
                - ``None``: return the positions of all nodes.

        Returns:
            pandas.Series/pandas.DataFrame:
                Series of ('x', 'y', 'z') if single node ID is

                passed as ``group``. Otherwise, a pandas.DataFrame of ('x', 'y', 'z') indexed
                by node IDs.
        """
        result = self.get(group=group, properties=[Node.X, Node.Y, Node.Z])
        return result.astype(float)

    def orientations(self, group=None):
        """Node orientation(s) as a pandas Series or DataFrame.

        Args:
            group (int/sequence/str/mapping/None): Which nodes will have their positions
                returned depends on the type of the ``group`` argument:

                - ``int``: return the orientation of a single node.
                - ``sequence``: return the orientations from a list of node IDs.
                - ``str``: return the orientations of nodes in a node set.
                - ``mapping``: return the orientations of nodes matching a properties filter.
                - ``None``: return the orientations of all nodes.

        Returns:
            numpy.ndarry/pandas.Series:
                A 3x3 rotation matrix if a single node ID is passed as ``group``.
                Otherwise a pandas Series with rotation matrices indexed by node IDs.
        """
        # need to keep this quaternion ordering for quaternion2mat (expects w, x, y , z)
        props = np.array([
            Node.ORIENTATION_W,
            Node.ORIENTATION_X,
            Node.ORIENTATION_Y,
            Node.ORIENTATION_Z
        ])
        props_mask = np.isin(props, list(self.property_names))
        orientation_count = np.count_nonzero(props_mask)
        if orientation_count == 4:
            trans = utils.quaternion2mat
        elif orientation_count in [1, 2, 3]:
            raise BluepySnapError(
                "Missing orientation fields. Should be 4 quaternions or euler angles or nothing")
        else:
            # need to keep this rotation_angle ordering for euler2mat (expects z, y, x)
            props = np.array([
                Node.ROTATION_ANGLE_Z,
                Node.ROTATION_ANGLE_Y,
                Node.ROTATION_ANGLE_X,
            ])
            props_mask = np.isin(props, list(self.property_names))
            trans = utils.euler2mat

        result = self.get(group=group, properties=props[props_mask])

        def _get_values(prop):
            """Retrieve prop from the result Dataframe/Series."""
            if isinstance(result, pd.Series):
                return [result.get(prop, 0)]
            return result.get(prop, np.zeros((result.shape[0],)))

        args = [_get_values(prop) for prop in props]
        if isinstance(group, six.integer_types + (np.integer,)):
            return trans(*args)[0]
        return pd.Series(trans(*args), index=result.index, name='orientation')

    def count(self, group=None):
        """Total number of nodes for a given node group.

        Args:
            group (int/sequence/str/mapping/None): Which nodes will have their positions
                returned depends on the type of the ``group`` argument:

                - ``int``: return the count of a single node.
                - ``sequence``: return the count of nodes from a list of node IDs.
                - ``str``: return the count of nodes in a node set.
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
            self._node_storage.circuit.config['components']['morphologies_dir'],
            self
        )
