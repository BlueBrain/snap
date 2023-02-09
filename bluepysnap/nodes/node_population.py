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

import numpy as np
import pandas as pd
from cached_property import cached_property
from more_itertools import first

from bluepysnap import query, utils
from bluepysnap.circuit_ids import CircuitNodeId, CircuitNodeIds
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.sonata_constants import DYNAMICS_PREFIX, ConstContainer, Node


class NodePopulation:
    """Node population access."""

    def __init__(self, circuit, population_name):
        """Initializes a NodePopulation object.

        Args:
            circuit (bluepysnap.Circuit): the circuit object containing the node population
            population_name (str): the name of the node population

        Returns:
            NodePopulation: A NodePopulation object.
        """
        self._circuit = circuit
        self.name = population_name

    @property
    def _node_sets(self):
        """Node sets defined for this node population."""
        return self._circuit.node_sets

    @cached_property
    def _data(self):
        """Collect data for the node population as a pandas.DataFrame."""
        nodes = self._population
        categoricals = nodes.enumeration_names

        _all = nodes.select_all()
        result = pd.DataFrame(index=np.arange(_all.flat_size))

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

    @property
    def _properties(self):
        return self._circuit.to_libsonata.node_population_properties(self.name)

    @property
    def _population(self):
        return self._circuit.to_libsonata.node_population(self.name)

    @cached_property
    def size(self):
        """Node population size."""
        return self._population.size

    @property
    def type(self):
        """Population type."""
        return self._properties.type

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
        return set(
            edge.name for edge in self._circuit.edges.values() if self.name == edge.source.name
        )

    def target_in_edges(self):
        """Set of edge population names that use this node population as target.

        Returns:
            set: a set containing the names of edge populations using this NodePopulation as
            target.
        """
        return set(
            edge.name for edge in self._circuit.edges.values() if self.name == edge.target.name
        )

    @property
    def population_config(self):
        """Access the configuration for the population.

        This configuration is extended with
        * 'components' of the circuit config
        * 'nodes_file': the path the h5 file containing the population.
        """
        return self._circuit.get_node_population_config(self.name)

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
            raise BluepySnapError(f"node ID not found: {node_id} in population '{self.name}'")

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
            raise BluepySnapError(
                f"All node IDs must be >= 0 and < {self._data.index.shape[0]} "
                f"for population '{self.name}'"
            )

    def _check_property(self, prop):
        """Check if a property exists inside the dataset."""
        if prop not in self.property_names:
            raise BluepySnapError(f"No such property: '{prop}'")

    def _get_node_set(self, node_set_name):
        """Returns the node set named 'node_set_name'."""
        if node_set_name not in self._node_sets:
            raise BluepySnapError(f"Undefined node set: '{node_set_name}'")
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
            result = self._node_ids_by_filter(
                queries=group, raise_missing_prop=raise_missing_property
            )
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
                except AttributeError as e:
                    raise BluepySnapError(
                        "All values from a list must be of type int or CircuitNodeId."
                    ) from e
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

            properties (list|str|None): If specified, return only the properties in the list.
                Otherwise return all properties.

        Returns:
            value/pandas.Series/pandas.DataFrame:
                The type of the returned object depends on the type of the input parameters,
                see the Examples for an explanation of the different cases.

        Notes:
            The NodePopulation.property_names function will give you all the usable properties
            for the `properties` argument.

        Examples:
            Considering a node population composed by 3 nodes (0, 1, 2) and 12 properties,
            the following examples show the types of the returned objects.

            - If ``group`` is a single node ID and ``properties`` a single property,
              returns a single scalar value.

                >>> result = my_node_population.get(group=0, properties=Cell.MTYPE)
                >>> type(result)
                str

            - If ``group`` is a single node ID and ``properties`` a list or None,
              returns a pandas Series indexed by the properties.

                >>> result = my_node_population.get(group=0)
                >>> type(result), result.shape
                (pandas.core.series.Series, (12,))

                >>> result = my_node_population.get(group=0, properties=[Cell.MTYPE])
                >>> type(result), result.shape
                (pandas.core.series.Series, (1,))

            - If ``group`` is anything other than a single node ID, and ``properties`` is a single
              property, returns a pandas Series indexed by node IDs.

                >>> result = my_node_population.get(properties=Cell.MTYPE)
                >>> type(result), result.shape
                (pandas.core.series.Series, (3,))

                >>> result = my_node_population.get(group=[0], properties=Cell.MTYPE)
                >>> type(result), result.shape
                (pandas.core.series.Series, (1,))

            - In all the other cases, returns a pandas DataFrame indexed by node IDs.

                >>> result = my_node_population.get()
                >>> type(result), result.shape
                (pandas.core.frame.DataFrame, (3, 12))

                >>> result = my_node_population.get(group=[0])
                >>> type(result), result.shape
                (pandas.core.frame.DataFrame, (1, 12))

                >>> result = my_node_population.get(properties=[Cell.MTYPE])
                >>> type(result), result.shape
                (pandas.core.frame.DataFrame, (3, 1))

                >>> result = my_node_population.get(group=[0], properties=[Cell.MTYPE])
                >>> type(result), result.shape
                (pandas.core.frame.DataFrame, (1, 1))
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
        props = np.array(
            [Node.ORIENTATION_W, Node.ORIENTATION_X, Node.ORIENTATION_Y, Node.ORIENTATION_Z]
        )
        props_mask = np.isin(props, list(self.property_names))
        orientation_count = np.count_nonzero(props_mask)
        if orientation_count == 4:
            trans = utils.quaternion2mat
        elif orientation_count in [1, 2, 3]:
            raise BluepySnapError(
                "Missing orientation fields. Should be 4 quaternions or euler angles or nothing"
            )
        else:
            # need to keep this rotation_angle ordering for euler2mat (expects z, y, x)
            props = np.array(
                [
                    Node.ROTATION_ANGLE_Z,
                    Node.ROTATION_ANGLE_Y,
                    Node.ROTATION_ANGLE_X,
                ]
            )
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
        return pd.Series(trans(*args), index=result.index, name="orientation")

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
            self._properties.morphologies_dir,
            self,
            alternate_morphologies=self._properties.alternate_morphology_formats,
        )

    @cached_property
    def models(self):
        """Access to node neuron models."""
        from bluepysnap.neuron_models import NeuronModelsHelper

        return NeuronModelsHelper(self._properties, self)

    @property
    def h5_filepath(self):
        """Get the H5 nodes file associated with population."""
        return self.population_config["nodes_file"]
