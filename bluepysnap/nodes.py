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
from copy import deepcopy

import libsonata
import numpy as np
import pandas as pd
import six

from cached_property import cached_property

from bluepysnap import utils
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.sonata_constants import (DYNAMICS_PREFIX, NODE_ID_KEY,
                                         POPULATION_KEY, Node, ConstContainer)


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
        self._circuit = circuit
        self._populations = {}

    @property
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
    def _node_sets(self):
        """Node sets defined for this node population."""
        return self._node_storage.circuit.node_sets

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
        """Check if a property exists inside the dataset."""
        if prop not in self.property_names:
            raise BluepySnapError("No such property: '%s'" % prop)

    def _get_node_set(self, node_set_name):
        """Returns the node set named 'node_set_name'."""
        if node_set_name not in self._node_sets:
            raise BluepySnapError("Undefined node set: '%s'" % node_set_name)
        return self._node_sets[node_set_name]

    def _positional_mask(self, node_ids):
        """Positional mask for the node IDs.

        Args:
            node_ids (None/numpy.ndarray): the ids array. If None all ids are selected.

        Examples:
            if the data set contains 5 nodes:
            _positional_mask([0,2]) --> [True, False, True, False, False]
        """
        if node_ids is None:
            return np.full(len(self._data), fill_value=True)
        mask = np.full(len(self._data), fill_value=False)
        mask[node_ids] = True
        return mask

    def _node_population_mask(self, queries):
        """Handle the population and node ID queries."""
        populations = queries.pop(POPULATION_KEY, None)
        if populations is not None and self.name not in set(utils.ensure_list(populations)):
            node_ids = []
        else:
            node_ids = queries.pop(NODE_ID_KEY, None)
        return queries, self._positional_mask(node_ids)

    def _properties_mask(self, queries):
        """Return mask of node IDs with rows matching `props` dict."""
        # pylint: disable=assignment-from-no-return
        unknown_props = set(queries) - set(self._data.columns) - {POPULATION_KEY, NODE_ID_KEY}
        if unknown_props:
            raise BluepySnapError("Unknown node properties: [{0}]".format(", ".join(unknown_props)))

        queries, mask = self._node_population_mask(queries)
        if not mask.any():
            # Avoid fail and/or processing time if wrong population or no nodes
            return mask

        for prop, values in six.iteritems(queries):
            prop = self._data[prop]
            if np.issubdtype(prop.dtype.type, np.floating):
                v1, v2 = values
                prop_mask = np.logical_and(prop >= v1, prop <= v2)
            elif isinstance(values, six.string_types) and values.startswith('regex:'):
                prop_mask = _complex_query(prop, {'$regex': values[6:]})
            elif isinstance(values, collections.Mapping):
                prop_mask = _complex_query(prop, values)
            else:
                prop_mask = np.in1d(prop, values)
            mask = np.logical_and(mask, prop_mask)
        return mask

    def _operator_mask(self, queries):
        """Handle the query operators '$or', '$and'."""
        if len(queries) == 0:
            return np.full(len(self._data), True)

        # will pop the population and or/and operators so need to copy
        queries = deepcopy(queries)
        first_key = list(queries)[0]
        if first_key == '$or':
            queries = queries.pop("$or")
            operator = np.logical_or
        elif first_key == '$and':
            queries = queries.pop("$and")
            operator = np.logical_and
        else:
            return self._properties_mask(queries)

        mask = np.full(len(self._data), first_key != "$or")
        for query in queries:
            mask = operator(mask, self._operator_mask(query))
        return mask

    def _node_ids_by_filter(self, queries):
        """Return node IDs if their properties match the `queries` dict.

        `props` values could be:
            pairs (range match for floating dtype fields)
            scalar or iterables (exact or "one of" match for other fields)

        You can use the special operators '$or' and '$and' also to combine different queries
        together.

        Examples:
            >>> _node_ids_by_filter({ Node.X: (0, 1), Node.MTYPE: 'L1_SLAC' })
            >>> _node_ids_by_filter({ Node.LAYER: [2, 3] })
            >>> _node_ids_by_filter({'$or': [{ Node.LAYER: [2, 3]},
            >>>                              { Node.X: (0, 1), Node.MTYPE: 'L1_SLAC' }]})

        """
        return self._data.index[self._operator_mask(queries)].values

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

        Examples:
            The available group parameter values:

            >>> nodes.ids(group=None)  #  returns all IDs
            >>> nodes.ids(group={})  #  returns all IDs
            >>> nodes.ids(group=1)  #  returns the single ID if present in population
            >>> nodes.ids(group=[1,2,3])  # returns list of IDs if all present in population
            >>> nodes.ids(group="node_set_name")  # returns list of IDs matching node set
            >>> nodes.ids(group={ Node.LAYER: 2})  # returns list of IDs matching layer==2
            >>> nodes.ids(group={ Node.LAYER: [2, 3]})  # returns list of IDs with layer in [2,3]
            >>> nodes.ids(group={ Node.X: (0, 1)})  # returns list of IDs with 0 < x < 1
            >>> # returns list of IDs matching one of the queries inside the 'or' list
            >>> nodes.ids(group={'$or': [{ Node.LAYER: [2, 3]},
            >>>                          { Node.X: (0, 1), Node.MTYPE: 'L1_SLAC' }]})
            >>> # returns list of IDs matching all the queries inside the 'and' list
            >>> nodes.ids(group={'$and': [{ Node.LAYER: [2, 3]},
            >>>                           { Node.X: (0, 1), Node.MTYPE: 'L1_SLAC' }]})
        """
        preserve_order = False
        if isinstance(group, six.string_types):
            group = self._get_node_set(group)

        if group is None:
            result = self._data.index.values
        elif isinstance(group, collections.Mapping):
            result = self._node_ids_by_filter(queries=group)
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

        result = np.array(result, dtype=np.int64)
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
            numpy.ndarray/pandas.Series:
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
