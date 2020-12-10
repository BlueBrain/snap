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

"""Edge population access."""
import inspect

import libsonata
import numpy as np
import pandas as pd
from cached_property import cached_property
from more_itertools import first

from bluepysnap.exceptions import BluepySnapError
from bluepysnap.circuit_ids import CircuitEdgeId, CircuitEdgeIds, CircuitNodeId, CircuitNodeIds
from bluepysnap.sonata_constants import DYNAMICS_PREFIX, Edge, ConstContainer
from bluepysnap import utils


class Edges:
    """The top level Edges accessor."""

    def __init__(self, circuit):
        """Initialize the top level Edges accessor."""
        self._circuit = circuit
        self._config = self._circuit.config['networks']['edges']
        self._populations = self._get_populations()

    def _get_populations(self):
        """Collect the different EdgePopulation."""
        res = {}
        for file_config in self._config:
            storage = EdgeStorage(file_config, self._circuit)
            for population in storage.population_names:  # pylint: disable=not-an-iterable
                if population in res:
                    raise BluepySnapError("Duplicated edge population: '%s'" % population)
                res[population] = storage.population(population)
        return res

    @cached_property
    def population_names(self):
        """Returns all the population names from the Circuit."""
        return sorted(self._populations)

    @cached_property
    def property_dtypes(self):
        """Returns all the property dtypes for the Circuit."""

        def _update(d, index, value):
            if d.setdefault(index, value) != value:
                raise BluepySnapError("Same property with different "
                                      "dtype. {}: {}!= {}".format(index, value, d[index]))

        res = dict()
        for pop in self.values():
            for varname, dtype in pop.property_dtypes.iteritems():
                _update(res, varname, dtype)
        return pd.Series(res)

    def keys(self):
        """Returns iterator on the population names.

        Made to simulate the behavior of a dict.keys().
        """
        return (name for name in self.population_names)

    def values(self):
        """Returns iterator on the EdgePopulations.

        Made to simulate the behavior of a dict.values().
        """
        return (self[name] for name in self.population_names)

    def items(self):
        """Returns iterator on the tuples (population name, EdgePopulations).

        Made to simulate the behavior of a dict.items().
        """
        return ((name, self[name]) for name in self.population_names)

    def __getitem__(self, population_name):
        """Access the EdgePopulation corresponding to the population 'population_name'."""
        try:
            return self._populations[population_name]
        except KeyError:
            raise BluepySnapError("{} not a edge population.".format(population_name))

    def __iter__(self):
        """Allows iteration over the different EdgePopulation."""
        return iter(self.keys())

    @cached_property
    def size(self):
        """Total number of edges inside the circuit."""
        return sum(pop.size for pop in self.values())

    @cached_property
    def property_names(self):
        """Returns all the properties present inside the circuit."""
        return set(prop for pop in self.values() for prop in pop.property_names)

    def _get_ids_from_pop(self, fun_to_apply, returned_ids_cls):
        """Get CircuitIds of class 'returned_ids_cls' for all populations using 'fun_to_apply'.

        Args:
            fun_to_apply (function): A function that returns the list of IDs for each population
                and the population containing these IDs.
            returned_ids_cls (CircuitNodeIds/CircuitEdgeIds): the class for the CircuitIds.

        Returns:
            CircuitNodeIds/CircuitEdgeIds: containing the IDs and the populations.
        """
        str_type = "<U{}".format(max(len(pop) for pop in self.population_names))
        ids = []
        populations = []
        for pop in self.values():
            pop_ids, name_ids = fun_to_apply(pop)
            pops = np.full_like(pop_ids, fill_value=name_ids, dtype=str_type)
            ids.append(pop_ids)
            populations.append(pops)
        ids = np.concatenate(ids).astype(np.int64)
        populations = np.concatenate(populations).astype(str_type)
        return returned_ids_cls.from_arrays(populations, ids)

    def ids(self, edge_ids):
        """Edge CircuitEdgeIds corresponding to edges ``edge_ids``.

        Args:
            edge_ids (int/CircuitEdgeId/CircuitEdgeIds/sequence): Which IDs will be
            returned depends on the type of the ``group`` argument:
                - ``CircuitEdgeId``: return the ID in a CircuitEdgeIds object.
                - ``CircuitEdgeIds``: return the IDs in a CircuitNodeIds object.
                - ``int``: returns a CircuitEdgeIds object containing the corresponding edge ID
                    for all populations.
                - ``sequence``: returns a CircuitNodeIds object containing the corresponding edge
                    IDs for all populations.

        Returns:
            CircuitEdgeIds: returns a CircuitEdgeIds containing all the edge IDs and the
                corresponding populations. For performance reasons we do not test if the edge ids
                are present or not in the circuit.

        Notes:
            This envision also the maybe future selection of edges on queries.
        """
        if isinstance(edge_ids, CircuitEdgeIds):
            diff = np.setdiff1d(edge_ids.get_populations(unique=True), self.population_names)
            if diff.size != 0:
                raise BluepySnapError("Population {} does not exist in the circuit.".format(diff))
        return self._get_ids_from_pop(lambda x: (x.ids(edge_ids), x.name), CircuitEdgeIds)

    def properties(self, edge_ids, properties):
        """Edge properties as pandas DataFrame.

        Args:
            edge_ids (int/CircuitEdgeId/CircuitEdgeIds/sequence): same as Edges.ids().
            properties (None/str/list): an edge property name or a list of edge property names.
                If set to None ids are returned.

        Returns:
            pandas.Series/pandas.DataFrame:
                A pandas Series indexed by edge IDs if ``properties`` is scalar.
                A pandas DataFrame indexed by edge IDs if ``properties`` is list.

        Notes:
            The Edges.property_names function will give you all the usable properties
            for the `properties` argument.
        """
        ids = self.ids(edge_ids)
        # TODO : remove this due to the addition of ids to the EdgePopulation
        if properties is None:
            return ids
        properties = utils.ensure_list(properties)

        unknown_props = set(properties) - self.property_names
        if unknown_props:
            raise BluepySnapError("Unknown properties required: {}".format(unknown_props))

        res = pd.DataFrame(index=ids.index, columns=properties)
        for name, pop in self.items():
            global_pop_ids = ids.filter_population(name)
            pop_ids = global_pop_ids.get_ids()
            pop_properties = set(properties) & pop.property_names
            # indices from EdgePopulation and Edge properties functions are different so I cannot
            # use a dataframe equal directly and properties have different types so cannot use a
            # multi dim numpy array
            for prop in pop_properties:
                res.loc[global_pop_ids.index, prop] = pop.properties(pop_ids, prop).to_numpy()
        return res.sort_index()

    def afferent_nodes(self, target, unique=True):
        """Get afferent CircuitNodeIDs for given target ``node_id``.

        Notes:
            Afferent nodes are nodes projecting an outgoing edge to one of the ``target`` node.

        Args:
            target (CircuitNodeIds/int/sequence/str/mapping/None): the target you want to resolve
            and use as target nodes.
            unique (bool): If ``True``, return only unique afferent node IDs.

        Returns:
            CircuitNodeIDs: Afferent CircuitNodeIDs for all the targets from all edge population.
        """
        target_ids = self._circuit.nodes.ids(target)
        result = self._get_ids_from_pop(lambda x: (x.afferent_nodes(target_ids), x.source.name),
                                        CircuitNodeIds)
        if unique:
            result.unique(inplace=True)
        return result

    def efferent_nodes(self, source, unique=True):
        """Get efferent node IDs for given source ``node_id``.

        Notes:
            Efferent nodes are nodes receiving an incoming edge from one of the ``source`` node.

        Args:
            source (CircuitNodeIds/int/sequence/str/mapping/None): the source you want to resolve
                and use as source nodes.
            unique (bool): If ``True``, return only unique afferent node IDs.

        Returns:
            numpy.ndarray: Efferent node IDs for all the sources.
        """
        source_ids = self._circuit.nodes.ids(source)
        result = self._get_ids_from_pop(lambda x: (x.efferent_nodes(source_ids), x.target.name),
                                        CircuitNodeIds)
        if unique:
            result.unique(inplace=True)
        return result

    def pathway_edges(self, source=None, target=None, properties=None):
        """Get edges corresponding to ``source`` -> ``target`` connections.

        Args:
            source: source node group
            target: target node group
            properties: None / edge property name / list of edge property names

        Returns:
            CircuitEdgeIDs, if ``properties`` is None;
            Pandas Series indexed by CircuitEdgeIDs if ``properties`` is string;
            Pandas DataFrame indexed by CircuitEdgeIDs if ``properties`` is list.
        """
        if source is None and target is None:
            raise BluepySnapError("Either `source` or `target` should be specified")

        source_ids = self._circuit.nodes.ids(source)
        target_ids = self._circuit.nodes.ids(target)

        result = self._get_ids_from_pop(lambda x: (x.pathway_edges(source_ids, target_ids), x.name),
                                        CircuitEdgeIds)

        if properties:
            result = self.properties(result, properties)
        return result

    def afferent_edges(self, node_id, properties=None):
        """Get afferent edges for given ``node_id``.

        Args:
            node_id (int): Target node ID.
            properties: An edge property name, a list of edge property names, or None.

        Returns:
            pandas.Series/pandas.DataFrame/list:
                A pandas Series indexed by edge ID if ``properties`` is a string.
                A pandas DataFrame indexed by edge ID if ``properties`` is a list.
                A list of edge IDs, if ``properties`` is None.
        """
        return self.pathway_edges(source=None, target=node_id, properties=properties)

    def efferent_edges(self, node_id, properties=None):
        """Get efferent edges for given ``node_id``.

        Args:
            node_id: source node ID
            properties: None / edge property name / list of edge property names

        Returns:
            List of edge IDs, if ``properties`` is None;
            Pandas Series indexed by edge IDs if ``properties`` is string;
            Pandas DataFrame indexed by edge IDs if ``properties`` is list.
        """
        return self.pathway_edges(source=node_id, target=None, properties=properties)

    def pair_edges(self, source_node_id, target_node_id, properties=None):
        """Get edges corresponding to ``source_node_id`` -> ``target_node_id`` connection.

        Args:
            source_node_id: source node ID
            target_node_id: target node ID
            properties: None / edge property name / list of edge property names

        Returns:
            List of edge IDs, if ``properties`` is None;
            Pandas Series indexed by edge IDs if ``properties`` is string;
            Pandas DataFrame indexed by edge IDs if ``properties`` is list.
        """
        return self.pathway_edges(
            source=source_node_id, target=target_node_id, properties=properties
        )

    @staticmethod
    def _add_circuit_ids(its, source, target):
        """Generator comprehension adding the CircuitIds to the iterator.

        Notes:
            Using closures or lambda functions would result in override functions and so the
            source and target would be the same for all the populations.
        """
        return ((CircuitNodeId(source, source_id), CircuitNodeId(target, target_id), count) for
                source_id, target_id, count in its)

    @staticmethod
    def _add_edge_ids(its, source, target, pop_name):
        """Generator comprehension adding the CircuitIds to the iterator."""
        return ((CircuitNodeId(source, source_id), CircuitNodeId(target, target_id),
                 CircuitEdgeIds.from_dict({pop_name: edge_id})) for source_id, target_id, edge_id in
                its)

    @staticmethod
    def _omit_edge_count(its, source, target):
        """Generator comprehension adding the CircuitIds to the iterator."""
        return ((CircuitNodeId(source, source_id), CircuitNodeId(target, target_id)) for
                source_id, target_id in its)

    def iter_connections(
            self, source=None, target=None, return_edge_ids=False,
            return_edge_count=False):
        """Iterate through ``source`` -> ``target`` connections.

        Args:
            source (CircuitNodeIds/int/sequence/str/mapping/None): source node group
            target (CircuitNodeIds/int/sequence/str/mapping/None): target node group
            return_edge_count: if True, edge count is added to yield result
            return_edge_ids: if True, edge ID list is added to yield result

        ``return_edge_count`` and ``return_edge_ids`` are mutually exclusive.

        Yields:
            (source_node_id, target_node_id, edge_ids) if return_edge_ids == True;
            (source_node_id, target_node_id, edge_count) if return_edge_count == True;
            (source_node_id, target_node_id) otherwise.
        """
        if return_edge_ids and return_edge_count:
            raise BluepySnapError(
                "`return_edge_count` and `return_edge_ids` are mutually exclusive"
            )
        for name, pop in self.items():
            it = pop.iter_connections(source=source, target=target,
                                      return_edge_ids=return_edge_ids,
                                      return_edge_count=return_edge_count)
            source_pop = pop.source.name
            target_pop = pop.target.name
            if return_edge_count:
                yield from self._add_circuit_ids(it, source_pop, target_pop)
            elif return_edge_ids:
                yield from self._add_edge_ids(it, source_pop, target_pop, name)
            else:
                yield from self._omit_edge_count(it, source_pop, target_pop)


class EdgeStorage:
    """Edge storage access."""

    def __init__(self, config, circuit):
        """Initializes a EdgeStorage object from a edge config and a Circuit.

        Args:
            config (dict): a edge config from the global circuit config
            circuit (bluepysnap.Circuit): the circuit object that contains the EdgePopulations
            from this storage.

        Returns:
            EdgeStorage: A EdgeStorage object.
        """
        self._h5_filepath = config['edges_file']
        self._csv_filepath = config['edge_types_file']
        self._circuit = circuit
        self._populations = {}

    @property
    def storage(self):
        """Access to the libsonata edge storage."""
        return libsonata.EdgeStorage(self._h5_filepath)

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
            self._populations[population_name] = EdgePopulation(self, population_name)
        return self._populations[population_name]


def _resolve_node_ids(nodes, group):
    """Node IDs corresponding to node group filter."""
    if group is None:
        return None
    return nodes.ids(group)


def _is_empty(xs):
    return (xs is not None) and (len(xs) == 0)


def _estimate_range_size(func, node_ids, n=3):
    """Median size of index second level for some node IDs from the provided list."""
    assert len(node_ids) > 0
    if len(node_ids) > n:
        node_ids = np.random.choice(node_ids, size=n, replace=False)
    return np.median([
        len(func(node_id).ranges) for node_id in node_ids
    ])


class EdgePopulation:
    """Edge population access."""

    def __init__(self, edge_storage, population_name):
        """Initializes a EdgePopulation object from a EdgeStorage and a population name.

        Args:
            edge_storage (EdgeStorage): the edge storage containing the edge population
            population_name (str): the name of the edge population

        Returns:
            EdgePopulation: An EdgePopulation object.
        """
        self._edge_storage = edge_storage
        self.name = population_name

    @cached_property
    def _population(self):
        return self._edge_storage.storage.open_population(self.name)

    @property
    def size(self):
        """Population size."""
        return self._population.size

    def _nodes(self, population_name):
        """Returns the NodePopulation corresponding to population."""
        result = self._edge_storage.circuit.nodes[population_name]
        return result

    @cached_property
    def source(self):
        """Source NodePopulation."""
        return self._nodes(self._population.source)

    @cached_property
    def target(self):
        """Target NodePopulation."""
        return self._nodes(self._population.target)

    @cached_property
    def _attribute_names(self):
        return set(self._population.attribute_names)

    @cached_property
    def _dynamics_params_names(self):
        return set(utils.add_dynamic_prefix(self._population.dynamics_attribute_names))

    @property
    def _topology_property_names(self):
        return {Edge.SOURCE_NODE_ID, Edge.TARGET_NODE_ID}

    @property
    def property_names(self):
        """Set of available edge properties.

        Notes:
            Properties are a combination of the group attributes, the dynamics_params and the
            topology properties.
        """
        return self._attribute_names | self._dynamics_params_names | self._topology_property_names

    @cached_property
    def property_dtypes(self):
        """Returns the dtypes of all the properties.

        Returns:
            pandas.Series: series indexed by field name with the corresponding dtype as value.
        """
        return self.properties([0], list(self.property_names)).dtypes.sort_index()

    def container_property_names(self, container):
        """Lists the ConstContainer properties shared with the EdgePopulation.

        Args:
            container (ConstContainer): a container class for edge properties.

        Returns:
            list: A list of strings corresponding to the properties that you can use from the
                container class

        Examples:
            >>> from bluepysnap.sonata_constants import Edge
            >>> print(my_edge_population.container_property_names(Edge))
            >>> ["AXONAL_DELAY", "SYN_WEIGHT"] # values you can use with my_edge_population
            >>> my_edge_population.property_values(Edge.AXONAL_DELAY)
            >>> my_edge_population.property_values(Edge.get("AXONAL_DELAY"))
        """
        if not inspect.isclass(container) or not issubclass(container, ConstContainer):
            raise BluepySnapError("'container' must be a subclass of ConstContainer")
        in_file = self.property_names
        return [k for k in container.key_set() if container.get(k) in in_file]

    def _get_property(self, prop, selection):
        if prop == Edge.SOURCE_NODE_ID:
            result = self._population.source_nodes(selection)
        elif prop == Edge.TARGET_NODE_ID:
            result = self._population.target_nodes(selection)
        elif prop in self._attribute_names:
            result = self._population.get_attribute(prop, selection)
        elif prop in self._dynamics_params_names:
            result = self._population.get_dynamics_attribute(
                prop.split(DYNAMICS_PREFIX)[1], selection)
        else:
            raise BluepySnapError("No such property: %s" % prop)
        return result

    def _get(self, selection, properties=None):
        """Get an array of edge IDs or DataFrame with edge properties."""
        edge_ids = np.asarray(selection.flatten(), dtype=np.int64)
        if properties is None:
            return edge_ids

        if utils.is_iterable(properties):
            if len(edge_ids) == 0:
                result = pd.DataFrame(columns=properties)
            else:
                result = pd.DataFrame(index=edge_ids)
                for p in properties:
                    result[p] = self._get_property(p, selection)
        else:
            if len(edge_ids) == 0:
                result = pd.Series(name=properties, dtype=np.float64)
            else:
                result = pd.Series(
                    self._get_property(properties, selection),
                    index=edge_ids,
                    name=properties
                )

        return result

    def ids(self, edge_ids):
        """Edge IDs corresponding to edges ``edge_ids``.

        Args:
            edge_ids (int/CircuitEdgeId/CircuitEdgeIds/sequence): Which IDs will be
                returned depends on the type of the ``group`` argument:

                - ``int``, ``CircuitEdgeId``: return a single edge ID.
                - ``CircuitEdgeIds`` return IDs of edges in an array.
                - ``sequence``: return IDs of edges in an array.

        Returns:
            numpy.array: A numpy array of IDs.
        """
        if isinstance(edge_ids, CircuitEdgeIds):
            result = edge_ids.filter_population(self.name).get_ids()
        elif isinstance(edge_ids, np.ndarray):
            result = edge_ids
        else:
            result = utils.ensure_list(edge_ids)
            # test if first value is a CircuitEdgeId if yes then all values must be CircuitEdgeId
            if isinstance(first(result, None), CircuitEdgeId):
                try:
                    result = [cid.id for cid in result if cid.population == self.name]
                except AttributeError:
                    raise BluepySnapError("All values from a list must be of type int or "
                                          "CircuitEdgeId.")
        return np.asarray(result)

    def properties(self, edge_ids, properties):
        """Edge properties as pandas DataFrame.

        Args:
            edge_ids (array-like): array-like of edge IDs
            properties (str/list): an edge property name or a list of edge property names

        Returns:
            pandas.Series/pandas.DataFrame:
                A pandas Series indexed by edge IDs if ``properties`` is scalar.
                A pandas DataFrame indexed by edge IDs if ``properties`` is list.

        Notes:
            The EdgePopulation.property_names function will give you all the usable properties
            for the `properties` argument.
        """
        edge_ids = self.ids(edge_ids)
        selection = libsonata.Selection(edge_ids)
        return self._get(selection, properties)

    def positions(self, edge_ids, side, kind):
        """Edge positions as a pandas DataFrame.

        Args:
            edge_ids (array-like): array-like of edge IDs
            side (str): ``afferent`` or ``efferent``
            kind (str): ``center`` or ``surface``

        Returns:
            Pandas Dataframe with ('x', 'y', 'z') columns indexed by edge IDs.
        """
        assert side in ('afferent', 'efferent')
        assert kind in ('center', 'surface')
        props = {
            '{side}_{kind}_{p}'.format(side=side, kind=kind, p=p): p
            for p in ['x', 'y', 'z']
        }
        result = self.properties(edge_ids, list(props))
        result.rename(columns=props, inplace=True)
        result.sort_index(axis=1, inplace=True)
        return result

    def afferent_nodes(self, target, unique=True):
        """Get afferent node IDs for given target ``node_id``.

        Notes:
            Afferent nodes are nodes projecting an outgoing edge to one of the ``target`` node.

        Args:
            target (CircuitNodeIds/int/sequence/str/mapping/None): the target you want to resolve
            and use as target nodes.
            unique (bool): If ``True``, return only unique afferent node IDs.

        Returns:
            numpy.ndarray: Afferent node IDs for all the targets.
        """
        if target is not None:
            selection = self._population.afferent_edges(
                _resolve_node_ids(self.target, target)
            )
        else:
            selection = self._population.select_all()
        result = self._population.source_nodes(selection)
        if unique:
            result = np.unique(result)
        return result

    def efferent_nodes(self, source, unique=True):
        """Get efferent node IDs for given source ``node_id``.

        Notes:
            Efferent nodes are nodes receiving an incoming edge from one of the ``source`` node.

        Args:
            source (CircuitNodeIds/int/sequence/str/mapping/None): the source you want to resolve
                and use as source nodes.
            unique (bool): If ``True``, return only unique afferent node IDs.

        Returns:
            numpy.ndarray: Efferent node IDs for all the sources.
        """
        if source is not None:
            selection = self._population.efferent_edges(
                _resolve_node_ids(self.source, source)
            )
        else:
            selection = self._population.select_all()
        result = self._population.target_nodes(selection)
        if unique:
            result = np.unique(result)
        return result

    def pathway_edges(self, source=None, target=None, properties=None):
        """Get edges corresponding to ``source`` -> ``target`` connections.

        Args:
            source (CircuitNodeIds/int/sequence/str/mapping/None): source node group
            target (CircuitNodeIds/int/sequence/str/mapping/None): target node group
            properties: None / edge property name / list of edge property names

        Returns:
            List of edge IDs, if ``properties`` is None;
            Pandas Series indexed by edge IDs if ``properties`` is string;
            Pandas DataFrame indexed by edge IDs if ``properties`` is list.
        """
        if source is None and target is None:
            raise BluepySnapError("Either `source` or `target` should be specified")

        source_node_ids = _resolve_node_ids(self.source, source)
        target_edge_ids = _resolve_node_ids(self.target, target)

        if source_node_ids is None:
            selection = self._population.afferent_edges(target_edge_ids)
        elif target_edge_ids is None:
            selection = self._population.efferent_edges(source_node_ids)
        else:
            selection = self._population.connecting_edges(source_node_ids, target_edge_ids)

        return self._get(selection, properties)

    def afferent_edges(self, node_id, properties=None):
        """Get afferent edges for given ``node_id``.

        Args:
            node_id (CircuitNodeIds/int/sequence/str/mapping/None) : Target node ID.
            properties: An edge property name, a list of edge property names, or None.

        Returns:
            pandas.Series/pandas.DataFrame/list:
                A pandas Series indexed by edge ID if ``properties`` is a string.
                A pandas DataFrame indexed by edge ID if ``properties`` is a list.
                A list of edge IDs, if ``properties`` is None.
        """
        return self.pathway_edges(source=None, target=node_id, properties=properties)

    def efferent_edges(self, node_id, properties=None):
        """Get efferent edges for given ``node_id``.

        Args:
            node_id (CircuitNodeIds/int/sequence/str/mapping/None): source node ID
            properties: None / edge property name / list of edge property names

        Returns:
            List of edge IDs, if ``properties`` is None;
            Pandas Series indexed by edge IDs if ``properties`` is string;
            Pandas DataFrame indexed by edge IDs if ``properties`` is list.
        """
        return self.pathway_edges(source=node_id, target=None, properties=properties)

    def pair_edges(self, source_node_id, target_node_id, properties=None):
        """Get edges corresponding to ``source_node_id`` -> ``target_node_id`` connection.

        Args:
            source_node_id (CircuitNodeIds/int/sequence/str/mapping/None): source node ID
            target_node_id (CircuitNodeIds/int/sequence/str/mapping/None): target node ID
            properties: None / edge property name / list of edge property names

        Returns:
            List of edge IDs, if ``properties`` is None;
            Pandas Series indexed by edge IDs if ``properties`` is string;
            Pandas DataFrame indexed by edge IDs if ``properties`` is list.
        """
        return self.pathway_edges(
            source=source_node_id, target=target_node_id, properties=properties
        )

    def _iter_connections(self, source_node_ids, target_node_ids, unique_node_ids, shuffle):
        """Iterate through `source_node_ids` -> `target_node_ids` connections."""
        # pylint: disable=too-many-branches,too-many-locals
        def _optimal_direction():
            """Choose between source and target node IDs for iterating."""
            if target_node_ids is None and source_node_ids is None:
                raise BluepySnapError("Either `source` or `target` should be specified")
            if source_node_ids is None:
                return 'target'
            if target_node_ids is None:
                return 'source'
            else:
                # Checking the indexing 'direction'. One direction has contiguous indices.
                range_size_source = _estimate_range_size(
                    self._population.efferent_edges, source_node_ids
                )
                range_size_target = _estimate_range_size(
                    self._population.afferent_edges, target_node_ids
                )
                return 'source' if (range_size_source < range_size_target) else 'target'

        if _is_empty(source_node_ids) or _is_empty(target_node_ids):
            return

        direction = _optimal_direction()
        if direction == 'target':
            primary_node_ids, secondary_node_ids = target_node_ids, source_node_ids
            get_connected_node_ids = self.afferent_nodes
        else:
            primary_node_ids, secondary_node_ids = source_node_ids, target_node_ids
            get_connected_node_ids = self.efferent_nodes

        primary_node_ids = np.unique(primary_node_ids)
        if shuffle:
            np.random.shuffle(primary_node_ids)

        if secondary_node_ids is not None:
            secondary_node_ids = np.unique(secondary_node_ids)

        secondary_node_ids_used = set()

        for key_node_id in primary_node_ids:
            connected_node_ids = get_connected_node_ids(key_node_id, unique=False)
            # [[secondary_node_id, count], ...]
            connected_node_ids_with_count = np.stack(
                np.unique(connected_node_ids, return_counts=True)
            ).transpose()
            # np.stack(uint64, int64) -> float64
            connected_node_ids_with_count = connected_node_ids_with_count.astype(np.uint32)
            if secondary_node_ids is not None:
                mask = np.in1d(connected_node_ids_with_count[:, 0],
                               secondary_node_ids, assume_unique=True)
                connected_node_ids_with_count = connected_node_ids_with_count[mask]
            if shuffle:
                np.random.shuffle(connected_node_ids_with_count)

            for conn_node_id, edge_count in connected_node_ids_with_count:
                if unique_node_ids and (conn_node_id in secondary_node_ids_used):
                    continue
                if direction == 'target':
                    yield conn_node_id, key_node_id, edge_count
                else:
                    yield key_node_id, conn_node_id, edge_count
                if unique_node_ids:
                    secondary_node_ids_used.add(conn_node_id)
                    break

    def iter_connections(
            self, source=None, target=None, unique_node_ids=False, shuffle=False,
            return_edge_ids=False, return_edge_count=False
    ):
        """Iterate through ``source`` -> ``target`` connections.

        Args:
            source (CircuitNodeIds/int/sequence/str/mapping/None): source node group
            target (CircuitNodeIds/int/sequence/str/mapping/None): target node group
            unique_node_ids: if True, no node ID will be used more than once as source or
                target for edges. Careful, this flag does not provide unique (source, target)
                pairs but unique node IDs.
            shuffle: if True, result order would be (somewhat) randomized
            return_edge_count: if True, edge count is added to yield result
            return_edge_ids: if True, edge ID list is added to yield result

        ``return_edge_count`` and ``return_edge_ids`` are mutually exclusive.

        Yields:
            (source_node_id, target_node_id, edge_ids) if return_edge_ids == True;
            (source_node_id, target_node_id, edge_count) if return_edge_count == True;
            (source_node_id, target_node_id) otherwise.
        """
        if return_edge_ids and return_edge_count:
            raise BluepySnapError(
                "`return_edge_count` and `return_edge_ids` are mutually exclusive"
            )

        source_node_ids = _resolve_node_ids(self.source, source)
        target_node_ids = _resolve_node_ids(self.target, target)

        it = self._iter_connections(source_node_ids, target_node_ids, unique_node_ids, shuffle)

        if return_edge_count:
            return it
        elif return_edge_ids:
            add_edge_ids = lambda x: (x[0], x[1], self.pair_edges(x[0], x[1]))
            return map(add_edge_ids, it)
        else:
            omit_edge_count = lambda x: x[:2]
            return map(omit_edge_count, it)
