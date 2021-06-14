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
import inspect
from collections.abc import Mapping, Sequence
from copy import deepcopy

from more_itertools import first
import libsonata
import numpy as np
import pandas as pd

from cached_property import cached_property

from bluepysnap import query
from bluepysnap.network import NetworkObject
from bluepysnap import utils
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.sonata_constants import (DYNAMICS_PREFIX, Node, ConstContainer)
from bluepysnap.circuit_ids import CircuitNodeId, CircuitNodeIds
from bluepysnap._doctools import AbstractDocSubstitutionMeta


class Nodes(NetworkObject, metaclass=AbstractDocSubstitutionMeta,
            source_word="NetworkObject", target_word="Node"):
    """The top level Nodes accessor."""

    def __init__(self, circuit):  # pylint: disable=useless-super-delegation
        """Initialize the top level Nodes accessor."""
        super().__init__(circuit)

    def _collect_populations(self):
        return self._get_populations(NodeStorage, self._config['networks']['nodes'])

    def property_values(self, prop):
        """Returns all the values for a given Nodes property."""
        return set(value for pop in self.values() if prop in pop.property_names for value in
                   pop.property_values(prop))

    def ids(self, group=None, sample=None, limit=None):
        """Returns the CircuitNodeIds corresponding to the nodes from ``group``.

        Args:
            group (CircuitNodeId/CircuitNodeIds/int/sequence/str/mapping/None): Which IDs will be
            returned depends on the type of the ``group`` argument:
                - ``CircuitNodeId``: return the ID in a CircuitNodeIds object if it belongs to
                    the circuit.
                - ``CircuitNodeIds``: return the IDs in a CircuitNodeIds object if they belong to
                    the circuit.
                - ``int``: if the node ID is present in all populations, returns a CircuitNodeIds
                    object containing the corresponding node ID for all populations.
                - ``sequence``: if all the values contained in the sequence are present in all
                    populations, returns a CircuitNodeIds object containing the corresponding node
                    IDs for all populations.
                - ``str``: use a node set name as input. Returns a CircuitNodeIds object containing
                    nodes selected by the node set.
                - ``mapping``: Returns a CircuitNodeIds object containing nodes matching a
                    properties filter.
                - ``None``: return all node IDs of the circuit in a CircuitNodeIds object.
            sample (int): If specified, randomly choose ``sample`` number of
                IDs from the match result. If the size of the sample is greater than
                the size of all the NodePopulations then all ids are taken and shuffled.
            limit (int): If specified, return the first ``limit`` number of
                IDs from the match result. If limit is greater than the size of all the populations,
                all node IDs are returned.

        Returns:
            CircuitNodeIds: returns a CircuitNodeIds containing all the node IDs and the
                corresponding populations. All the explicitly requested IDs must be present inside
                the circuit.

        Raises:
            BluepySnapError: when a population from a CircuitNodeIds is not present in the circuit.
            BluepySnapError: when an id query via a int, sequence, or CircuitNodeIds is not present
                in the circuit.

        Examples:
            The available group parameter values (example with 2 node populations pop1 and pop2):

            >>> nodes = circuit.nodes
            >>> nodes.ids(group=None)  #  returns all CircuitNodeIds from the circuit
            >>> node_ids = CircuitNodeIds.from_arrays(["pop1", "pop2"], [1, 3])
            >>> nodes.ids(group=node_ids)  #  returns ID 1 from pop1 and ID 3 from pop2
            >>> nodes.ids(group=0)  #  returns CircuitNodeIds 0 from pop1 and pop2
            >>> nodes.ids(group=[0, 1])  #  returns CircuitNodeIds 0 and 1 from pop1 and pop2
            >>> nodes.ids(group="node_set_name")  # returns CircuitNodeIds matching node set
            >>> nodes.ids(group={Node.LAYER: 2})  # returns CircuitNodeIds matching layer==2
            >>> nodes.ids(group={Node.LAYER: [2, 3]})  # returns CircuitNodeIds with layer in [2,3]
            >>> nodes.ids(group={Node.X: (0, 1)})  # returns CircuitNodeIds with 0 < x < 1
            >>> # returns CircuitNodeIds matching one of the queries inside the 'or' list
            >>> nodes.ids(group={'$or': [{ Node.LAYER: [2, 3]},
            >>>                          { Node.X: (0, 1), Node.MTYPE: 'L1_SLAC' }]})
            >>> # returns CircuitNodeIds matching all the queries inside the 'and' list
            >>> nodes.ids(group={'$and': [{ Node.LAYER: [2, 3]},
            >>>                           { Node.X: (0, 1), Node.MTYPE: 'L1_SLAC' }]})
        """
        if isinstance(group, CircuitNodeIds):
            diff = np.setdiff1d(group.get_populations(unique=True), self.population_names)
            if diff.size != 0:
                raise BluepySnapError("Population {} does not exist in the circuit.".format(diff))

        fun = lambda x: (x.ids(group, raise_missing_property=False), x.name)
        return self._get_ids_from_pop(fun, CircuitNodeIds, sample=sample, limit=limit)

    def get(self, group=None, properties=None):   # pylint: disable=arguments-differ
        """Node properties as a pandas DataFrame.

        Args:
            group (CircuitNodeIds/int/sequence/str/mapping/None): Which nodes will have their
                properties returned depends on the type of the ``group`` argument:
                See :py:class:`~bluepysnap.nodes.Nodes.ids`.

            properties (str/list): If specified, return only the properties in the list.
                Otherwise return all properties.

        Returns:
            pandas.DataFrame: Return a pandas DataFrame indexed by NodeCircuitIds containing the
                properties from ``properties``.

        Notes:
            The NodePopulation.property_names function will give you all the usable properties
            for the `properties` argument.
        """
        if properties is None:
            properties = self.property_names
        return super().get(group, properties)


class NodeStorage:
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
    def h5_filepath(self):
        """Returns the filepath of the Storage."""
        return self._h5_filepath

    @property
    def csv_filepath(self):
        """Returns the csv filepath of the Storage."""
        return self._csv_filepath

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
        categoricals = nodes.enumeration_names

        node_count = nodes.size
        result = pd.DataFrame(index=np.arange(node_count))

        _all = libsonata.Selection([(0, node_count)])
        for attr in sorted(nodes.attribute_names):
            if attr in categoricals:
                enumeration = np.asarray(nodes.get_enumeration(attr, _all))
                values = np.asarray(nodes.enumeration_values(attr))
                # if the size of `values` is large enough compared to `enumeration`, not using
                # categorical reduces the memory usage.
                if values.shape[0] < 0.5 * enumeration.shape[0]:
                    result[attr] = pd.Categorical.from_codes(enumeration, categories=values)
                else:
                    result[attr] = values[enumeration]
            else:
                result[attr] = nodes.get_attribute(attr, _all)
        for attr in sorted(utils.add_dynamic_prefix(nodes.dynamics_attribute_names)):
            result[attr] = nodes.get_dynamics_attribute(attr.split(DYNAMICS_PREFIX)[1], _all)
        return result


class NodePopulation:
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

    def source_in_edges(self):
        """Set of edge population names that use this node population as source.

        Returns:
            set: a set containing the names of edge populations using this NodePopulation as
            source.
        """
        return set(edge.name for edge in self._node_storage.circuit.edges.values() if
                   self.name == edge.source.name)

    def target_in_edges(self):
        """Set of edge population names that use this node population as target.

        Returns:
            set: a set containing the names of edge populations using this NodePopulation as
            target.
        """
        return set(edge.name for edge in self._node_storage.circuit.edges.values() if
                   self.name == edge.target.name)

    @property
    def property_names(self):
        """Set of available node properties.

        Notes:
            Properties are a combination of the group attributes and the dynamics_params.
        """
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

    def property_values(self, prop, is_present=False):
        """Set of values for a given property.

        Args:
           prop (str): Name of the property to retrieve.
           is_present (bool): if the field is categorical it forces the values to be actually
           present inside the dataset. (see: Notes for more information).

        Returns:
            set: A set of the unique values of the property in the node population.

        Notes:
            For categorical fields returning 'unique' values can be confusing, see:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html
            in #working-with-categories.
            The is_present argument forces the unique() even on the categorical fields.
        """
        res = self.get(properties=prop)
        if pd.api.types.is_categorical_dtype(res) and not is_present:
            return set(res.cat.categories)
        return set(res.unique())

    @cached_property
    def property_dtypes(self):
        """Returns the dtypes of all the properties.

        Returns:
            pandas.Series: series indexed by field name with the corresponding dtype as value.
        """
        return self._data.dtypes.sort_index()

    def _check_id(self, node_id):
        """Check that single node ID belongs to the circuit."""
        if node_id not in self._data.index:
            raise BluepySnapError("node ID not found: {} in population"
                                  " '{}'".format(node_id, self.name))

    def _check_ids(self, node_ids):
        """Check that node IDs belong to the circuit."""
        if len(node_ids) == 0:
            return
        # use the function with better performance for arrays or lists
        if isinstance(node_ids, np.ndarray):
            max_id = node_ids.max()
            min_id = node_ids.min()
        else:
            max_id = max(node_ids)
            min_id = min(node_ids)
        if min_id < 0 or max_id >= self._data.index.shape[0]:
            raise BluepySnapError("All node IDs must be >= 0 and < {} for population "
                                  "'{}'".format(self._data.index.shape[0], self.name))

    def _check_property(self, prop):
        """Check if a property exists inside the dataset."""
        if prop not in self.property_names:
            raise BluepySnapError("No such property: '%s'" % prop)

    def _get_node_set(self, node_set_name):
        """Returns the node set named 'node_set_name'."""
        if node_set_name not in self._node_sets:
            raise BluepySnapError("Undefined node set: '%s'" % node_set_name)
        return self._node_sets[node_set_name]

    def _resolve_nodesets(self, queries):
        def _resolve(queries, queries_key):
            if queries_key == query.NODE_SET_KEY:
                if query.AND_KEY not in queries:
                    queries[query.AND_KEY] = []
                queries[query.AND_KEY].append(self._get_node_set(queries[queries_key]))
                del queries[queries_key]

        resolved_queries = deepcopy(queries)
        query.traverse_queries_bottom_up(resolved_queries, _resolve)
        return resolved_queries

    def _node_ids_by_filter(self, queries, raise_missing_prop):
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
        queries = self._resolve_nodesets(queries)
        if raise_missing_prop:
            properties = query.get_properties(queries)
            if not properties.issubset(self._data.columns):
                unknown_props = properties - set(self._data.columns)
                raise BluepySnapError(f"Unknown node properties: {unknown_props}")
        idx = query.resolve_ids(self._data, self.name, queries)
        return self._data.index[idx].values

    def ids(self, group=None, limit=None, sample=None, raise_missing_property=True):
        """Node IDs corresponding to node ``group``.

        Args:
            group (int/CircuitNodeId/CircuitNodeIds/sequence/str/mapping/None): Which IDs will be
                returned depends on the type of the ``group`` argument:

                - ``int``, ``CircuitNodeId``: return a single node ID if it belongs to the circuit.
                - ``CircuitNodeIds`` return IDs of nodes from the node population in an array.
                - ``sequence``: return IDs of nodes in an array.
                - ``str``: return IDs of nodes in a node set.
                - ``mapping``: return IDs of nodes matching a properties filter.
                - ``None``: return all node IDs.

                If ``group`` is a ``sequence``, the order of results is preserved.
                Otherwise the result is sorted and contains no duplicates.

            sample (int): If specified, randomly choose ``sample`` number of
                IDs from the match result. If the size of the sample is greater than
                the size of the NodePopulation then all ids are taken and shuffled.

            limit (int): If specified, return the first ``limit`` number of
                IDs from the match result. If limit is greater than the size of the population
                all node IDs are returned.

            raise_missing_property (bool): if True, raises if a property is not listed in this
                population. Otherwise the ids are just not selected if a property is missing.

        Returns:
            numpy.array: A numpy array of IDs.

        Examples:
            The available group parameter values:

            >>> nodes.ids(group=None)  #  returns all IDs
            >>> nodes.ids(group={})  #  returns all IDs
            >>> nodes.ids(group=1)  #  returns the single ID if present in population
            >>> #  returns the single ID if present in population and the circuit id population
            >>> #  corresponds to nodes.name
            >>> nodes.ids(group=CircuitNodeId('pop', 1))
            >>> nodes.ids(group=[1,2,3])  # returns list of IDs if all present in population
            >>> #  returns list of IDs if all present in population
            >>> nodes.ids(group=CircuitNodeIds.from_dict({"pop": [0, 1,2]}))
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
        # pylint: disable=too-many-branches
        preserve_order = False
        if isinstance(group, str):
            group = self._get_node_set(group)
        elif isinstance(group, CircuitNodeIds):
            group = group.filter_population(self.name).get_ids()

        if group is None:
            result = self._data.index.values
        elif isinstance(group, Mapping):
            result = self._node_ids_by_filter(queries=group,
                                              raise_missing_prop=raise_missing_property)
        elif isinstance(group, np.ndarray):
            result = group
            self._check_ids(result)
            preserve_order = True
        else:
            result = utils.ensure_list(group)
            # test if first value is a CircuitNodeId all values are all CircuitNodeId
            if isinstance(first(result, None), CircuitNodeId):
                try:
                    result = [cid.id for cid in result if cid.population == self.name]
                except AttributeError:
                    raise BluepySnapError("All values from a list must be of type int or "
                                          "CircuitNodeId.")
            self._check_ids(result)
            preserve_order = isinstance(group, Sequence)

        if sample is not None:
            if len(result) > 0:
                result = np.random.choice(result, min(sample, len(result)), replace=False)
            preserve_order = False
        if limit is not None:
            result = result[:limit]

        result = utils.ensure_ids(result)
        if preserve_order:
            return result
        else:
            return np.unique(result)

    def get(self, group=None, properties=None):
        """Node properties as a pandas Series or DataFrame.

        Args:
            group (int/CircuitNodeId/CircuitNodeIds/sequence/str/mapping/None): Which nodes will
                have their properties returned depends on the type of the ``group`` argument:

                - ``int``, ``CircuitNodeId``: return the properties of a single node.
                - ``CircuitNodeIds`` return the properties from a NodeCircuitNodeIds.
                - ``sequence``: return the properties from a list of node.
                - ``str``: return the properties of nodes in a node set.
                - ``mapping``: return the properties of nodes matching a properties filter.
                - ``None``: return the properties of all nodes.

            properties (list): If specified, return only the properties in the list.
                Otherwise return all properties.

        Returns:
            value/pandas.Series/pandas.DataFrame:
                If single node ID is passed as ``group`` and single property as properties returns
                a single value. If single node ID is passed as ``group`` and list as property
                returns a pandas Series. Otherwise return a pandas DataFrame indexed by node IDs.

        Notes:
            The NodePopulation.property_names function will give you all the usable properties
            for the `properties` argument.
        """
        result = self._data
        if properties is not None:
            for p in utils.ensure_list(properties):
                self._check_property(p)
            result = result[properties]

        if group is not None:
            if isinstance(group, (int, np.integer)):
                self._check_id(group)
            elif isinstance(group, CircuitNodeId):
                group = self.ids(group)[0]
            else:
                group = self.ids(group)
            result = result.loc[group]

        return result

    def positions(self, group=None):
        """Node position(s) as pandas Series or DataFrame.

        Args:
            group (int/CircuitNodeId/CircuitNodeIds/sequence/str/mapping/None): Which nodes will
                have their positions returned depends on the type of the ``group`` argument:

                - ``int``, ``CircuitNodeId``: return the position of a single node.
                - ``CircuitNodeIds`` return the position from a NodeCircuitNodeIds.
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
        """Node orientation(s) as a pandas numpy array or pandas Series.

        Args:
            group (int/CircuitNodeId/CircuitNodeIds/sequence/str/mapping/None): Which nodes will
                have their positions returned depends on the type of the ``group`` argument:

                - ``int``, ``CircuitNodeId``: return the orientation of a single node.
                - ``CircuitNodeIds`` return the orientation from a NodeCircuitNodeIds.
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
        if utils.is_node_id(group):
            return trans(*args)[0]
        return pd.Series(trans(*args), index=result.index, name='orientation')

    def count(self, group=None):
        """Total number of nodes for a given node group.

        Args:
            group (int/CircuitNodeId/CircuitNodeIds/sequence/str/mapping/None): Which nodes will
                have their positions returned depends on the type of the ``group`` argument:

                - ``int``, ``CircuitNodeId``: return the count of a single node.
                - ``CircuitNodeIds`` return the count of nodes from a NodeCircuitNodeIds.
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

    @cached_property
    def models(self):
        """Access to node neuron models."""
        from bluepysnap.neuron_models import NeuronModelsHelper
        return NeuronModelsHelper(self._node_storage.circuit.config['components'], self)
