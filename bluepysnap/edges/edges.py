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

"""Edges access."""

import numpy as np

from bluepysnap._doctools import AbstractDocSubstitutionMeta
from bluepysnap.circuit_ids import CircuitEdgeIds, CircuitNodeIds
from bluepysnap.edges.edge_population import EdgePopulation
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.network import NetworkObject


class Edges(
    NetworkObject,
    metaclass=AbstractDocSubstitutionMeta,
    source_word="NetworkObject",
    target_word="Edge",
):
    """The top level Edges accessor."""

    _population_class = EdgePopulation

    def __init__(self, circuit):  # pylint: disable=useless-super-delegation
        """Initialize the top level Edges accessor."""
        super().__init__(circuit)

    @property
    def population_names(self):
        """Defines all sorted edge population names from the Circuit."""
        return sorted(self._circuit.to_libsonata.edge_populations)

    def ids(self, group=None, sample=None, limit=None):
        """Edge CircuitEdgeIds corresponding to edges ``edge_ids``.

        Args:
            group (None/int/CircuitEdgeId/CircuitEdgeIds/sequence): Which IDs will be
                returned depends on the type of the ``group`` argument:

                - ``None``: return all CircuitEdgeIds.
                - ``CircuitEdgeId``: return the ID in a CircuitEdgeIds object.
                - ``CircuitEdgeIds``: return the IDs in a CircuitNodeIds object.
                - ``int``: returns a CircuitEdgeIds object containing the corresponding edge ID
                  for all populations.
                - ``sequence``: returns a CircuitEdgeIds object containing the corresponding edge
                  IDs for all populations.
            sample (int): If specified, randomly choose ``sample`` number of
                IDs from the match result. If the size of the sample is greater than
                the size of all the EdgePopulations then all ids are taken and shuffled.
            limit (int): If specified, return the first ``limit`` number of
                IDs from the match result. If limit is greater than the size of all the populations
                all node IDs are returned.

        Returns:
            CircuitEdgeIds: returns a CircuitEdgeIds containing all the edge IDs and the
            corresponding populations. For performance reasons we do not test if the edge ids
            are present or not in the circuit.

        Notes:
            This envision also the maybe future selection of edges on queries.
        """
        if isinstance(group, CircuitEdgeIds):
            diff = np.setdiff1d(group.get_populations(unique=True), self.population_names)
            if diff.size != 0:
                raise BluepySnapError(f"Population {diff} does not exist in the circuit.")
        fun = lambda x: (x.ids(group), x.name)
        return self._get_ids_from_pop(fun, CircuitEdgeIds, sample=sample, limit=limit)

    def get(self, edge_ids=None, properties=None):  # pylint: disable=arguments-renamed
        """Edge properties by iterating populations.

        Args:
            edge_ids (int/CircuitEdgeId/CircuitEdgeIds/sequence): same as Edges.ids().
            properties (None/str/list): an edge property name or a list of edge property names.
                If set to None ids are returned.

        Returns:
            generator: yields tuples of ``(<population_name>, pandas.DataFrame)``:
                - DataFrame indexed by CircuitEdgeIds containing the properties from ``properties``.

        Notes:
            The Edges.property_names function will give you all the usable properties
            for the `properties` argument.
        """
        if edge_ids is None:
            raise BluepySnapError("You need to set edge_ids in get.")
        if properties is None:
            return edge_ids
        return super().get(edge_ids, properties)

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
        result = self._get_ids_from_pop(
            lambda x: (x.afferent_nodes(target_ids), x.source.name), CircuitNodeIds
        )
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
        result = self._get_ids_from_pop(
            lambda x: (x.efferent_nodes(source_ids), x.target.name), CircuitNodeIds
        )
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
            - CircuitEdgeIDs, if ``properties`` is None;
            - Pandas Series indexed by CircuitEdgeIDs if ``properties`` is string;
            - Pandas DataFrame indexed by CircuitEdgeIDs if ``properties`` is list.
        """
        if source is None and target is None:
            raise BluepySnapError("Either `source` or `target` should be specified")

        source_ids = self._circuit.nodes.ids(source)
        target_ids = self._circuit.nodes.ids(target)

        result = self._get_ids_from_pop(
            lambda x: (x.pathway_edges(source_ids, target_ids), x.name), CircuitEdgeIds
        )

        if properties:
            return self.get(result, properties)
        return result

    def afferent_edges(self, node_id, properties=None):
        """Get afferent edges for given ``node_id``.

        Args:
            node_id (int): Target node ID.
            properties: An edge property name, a list of edge property names, or None.

        Returns:
            pandas.Series/pandas.DataFrame/list:
                - A pandas Series indexed by edge ID if ``properties`` is a string.
                - A pandas DataFrame indexed by edge ID if ``properties`` is a list.
                - A list of edge IDs, if ``properties`` is None.
        """
        return self.pathway_edges(source=None, target=node_id, properties=properties)

    def efferent_edges(self, node_id, properties=None):
        """Get efferent edges for given ``node_id``.

        Args:
            node_id: source node ID
            properties: None / edge property name / list of edge property names

        Returns:
            - List of edge IDs, if ``properties`` is None;
            - Pandas Series indexed by edge IDs if ``properties`` is string;
            - Pandas DataFrame indexed by edge IDs if ``properties`` is list.
        """
        return self.pathway_edges(source=node_id, target=None, properties=properties)

    def pair_edges(self, source_node_id, target_node_id, properties=None):
        """Get edges corresponding to ``source_node_id`` -> ``target_node_id`` connection.

        Args:
            source_node_id: source node ID
            target_node_id: target node ID
            properties: None / edge property name / list of edge property names

        Returns:
            - List of edge IDs, if ``properties`` is None;
            - Pandas Series indexed by edge IDs if ``properties`` is string;
            - Pandas DataFrame indexed by edge IDs if ``properties`` is list.
        """
        return self.pathway_edges(
            source=source_node_id, target=target_node_id, properties=properties
        )

    def iter_connections(
        self, source=None, target=None, return_edge_ids=False, return_edge_count=False
    ):
        """Iterate through ``source`` -> ``target`` connections.

        Args:
            source (CircuitNodeIds/int/sequence/str/mapping/None): source node group
            target (CircuitNodeIds/int/sequence/str/mapping/None): target node group
            return_edge_count: if True, edge count is added to yield result
            return_edge_ids: if True, edge ID list is added to yield result

        ``return_edge_count`` and ``return_edge_ids`` are mutually exclusive.

        Yields:
            - (source_node_id, target_node_id, edge_ids) if ``return_edge_ids`` is True;
            - (source_node_id, target_node_id, edge_count) if ``return_edge_count`` is True;
            - (source_node_id, target_node_id) otherwise.
        """
        if return_edge_ids and return_edge_count:
            raise BluepySnapError(
                "`return_edge_count` and `return_edge_ids` are mutually exclusive"
            )
        for pop in self.values():
            yield from pop.iter_connections(
                source=source,
                target=target,
                return_edge_ids=return_edge_ids,
                return_edge_count=return_edge_count,
            )

    def __getstate__(self):
        """Make Edges pickle-able, without storing state of caches."""
        return self._circuit

    def __setstate__(self, state):
        """Load from pickle state."""
        self.__init__(state)
