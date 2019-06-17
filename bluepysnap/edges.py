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

"""
Edge population access.
"""

from builtins import map

import collections

import libsonata
import numpy as np
import pandas as pd

from cached_property import cached_property

from bluepysnap.exceptions import BlueSnapError
from bluepysnap.utils import is_iterable


SOURCE_NODE_ID = "@source_node"
TARGET_NODE_ID = "@target_node"


def _get_population_name(h5_filepath):
    populations = libsonata.EdgeStorage(h5_filepath).population_names
    if len(populations) != 1:
        raise BlueSnapError(
            "Only single-population node collections are supported (found: %d)" % len(populations)
        )
    return list(populations)[0]


def _resolve_node_ids(nodes, group):
    """ Node IDs corresponding to node group filter. """
    if group is None:
        return None
    return nodes.ids(group)


def _is_empty(xs):
    return (xs is not None) and (len(xs) == 0)


def _estimate_range_size(func, node_ids, n=3):
    """ Median size of index second level for some node IDs from the provided list. """
    assert len(node_ids) > 0
    if len(node_ids) > n:
        node_ids = np.random.choice(node_ids, size=n, replace=False)
    return np.median([
        len(func(node_id).ranges) for node_id in node_ids
    ])


class EdgePopulation(object):
    """ Edge population access. """
    def __init__(self, config, circuit):
        self._h5_filepath = config['edges_file']
        self._csv_filepath = config['edge_types_file']
        self._circuit = circuit

    @cached_property
    def _population(self):
        return libsonata.EdgePopulation(self._h5_filepath, self._csv_filepath or '', self.name)

    @cached_property
    def name(self):
        """ Population name. """
        return _get_population_name(self._h5_filepath)

    @property
    def size(self):
        """ Population size. """
        return self._population.size

    def _nodes(self, population):
        nodes = self._circuit.nodes

        result = None
        if isinstance(nodes, collections.Mapping):
            result = nodes.get(population)
        elif nodes.name == population:
            result = nodes

        if result is None:
            raise BlueSnapError("Undefined node population: '%s'" % population)

        return result

    @property
    def source(self):
        """ Source NodePopulation. """
        return self._nodes(self._population.source)

    @cached_property
    def target(self):
        """ Target NodePopulation. """
        return self._nodes(self._population.target)

    @property
    def property_names(self):
        """ Set of available edge properties. """
        return set(self._population.attribute_names)

    def _get_property(self, prop, selection):
        if prop == SOURCE_NODE_ID:
            result = self._population.source_nodes(selection)
        elif prop == TARGET_NODE_ID:
            result = self._population.target_nodes(selection)
        elif prop in self.property_names:
            result = self._population.get_attribute(prop, selection)
        else:
            raise BlueSnapError("No such property: %s" % prop)
        return result

    def _get(self, selection, properties=None):
        """ Get an array of edge IDs or DataFrame with edge properties. """
        edge_ids = selection.flatten()

        if properties is None:
            return edge_ids

        if is_iterable(properties):
            if len(edge_ids) == 0:
                result = pd.DataFrame(columns=properties)
            else:
                result = pd.DataFrame(index=edge_ids)
                for p in properties:
                    result[p] = self._get_property(p, selection)
        else:
            if len(edge_ids) == 0:
                result = pd.Series(name=properties)
            else:
                result = pd.Series(
                    self._get_property(properties, selection),
                    index=edge_ids,
                    name=properties
                )

        return result

    def properties(self, edge_ids, properties):
        """
        Edge properties as pandas DataFrame.

        Args:
            edge_ids: array-like of edge IDs
            properties: edge property name | list of edge property names

        Returns:
            Pandas Series indexed by edge IDs if `properties` is scalar;
            Pandas DataFrame indexed by edge IDs if `properties` is list.
        """
        selection = libsonata.Selection(edge_ids)
        return self._get(selection, properties)

    def positions(self, edge_ids, side, kind):
        """
        Edge positions as pandas DataFrame.

        Args:
            edge_ids: array-like of edge IDs
            side: 'afferent' | 'efferent'
            kind: 'center' | 'surface'

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

    def afferent_nodes(self, node_id, unique=True):
        """
        Get afferent node IDs for given target `node_id`.

        Args:
            node_id: target node ID

        Returns:
            Array of source node IDs.
            By default, the array is uniq-ed and sorted.
        """
        selection = self._population.afferent_edges(
            _resolve_node_ids(self.target, node_id)
        )
        result = self._population.source_nodes(selection)
        if unique:
            result = np.unique(result)
        return result

    def efferent_nodes(self, node_id, unique=True):
        """
        Get efferent node IDs for given source `node_id`.

        Args:
            node_id: source node ID

        Returns:
            Array of target node IDs.
            By default, the array is uniq-ed and sorted.
        """
        selection = self._population.efferent_edges(
            _resolve_node_ids(self.source, node_id)
        )
        result = self._population.target_nodes(selection)
        if unique:
            result = np.unique(result)
        return result

    def afferent_edges(self, node_id, properties=None):
        """
        Get afferent edges for given `node_id`.

        Args:
            node_id: target node ID
            properties: None / edge property name / list of edge property names

        Returns:
            List of edge IDs, if `properties` is None;
            Pandas Series indexed by edge IDs if `properties` is string;
            Pandas DataFrame indexed by edge IDs if `properties` is list.
        """
        return self.pathway_edges(source=None, target=node_id, properties=properties)

    def efferent_edges(self, node_id, properties=None):
        """
        Get efferent edges for given `node_id`.

        Args:
            node_id: source node ID
            properties: None / edge property name / list of edge property names

        Returns:
            List of edge IDs, if `properties` is None;
            Pandas Series indexed by edge IDs if `properties` is string;
            Pandas DataFrame indexed by edge IDs if `properties` is list.
        """
        return self.pathway_edges(source=node_id, target=None, properties=properties)

    def pair_edges(self, source_node_id, target_node_id, properties=None):
        """
        Get edges corresponding to `source_node_id` -> `target_node_id` connection.

        Args:
            source_node_id: source node ID
            target_node_id: target node ID
            properties: None / edge property name / list of edge property names

        Returns:
            List of edge IDs, if `properties` is None;
            Pandas Series indexed by edge IDs if `properties` is string;
            Pandas DataFrame indexed by edge IDs if `properties` is list.
        """
        return self.pathway_edges(
            source=source_node_id, target=target_node_id, properties=properties
        )

    def pathway_edges(self, source=None, target=None, properties=None):
        """
        Get edges corresponding to `source` -> `target` connections.

        Args:
            source: source node group
            target: target node group
            properties: None / edge property name / list of edge property names

        Returns:
            List of edge IDs, if `properties` is None;
            Pandas Series indexed by edge IDs if `properties` is string;
            Pandas DataFrame indexed by edge IDs if `properties` is list.
        """
        if source is None and target is None:
            raise BlueSnapError("Either `source` or `target` should be specified")

        source_node_ids = _resolve_node_ids(self.source, source)
        target_edge_ids = _resolve_node_ids(self.target, target)

        if source_node_ids is None:
            selection = self._population.afferent_edges(target_edge_ids)
        elif target_edge_ids is None:
            selection = self._population.efferent_edges(source_node_ids)
        else:
            selection = self._population.connecting_edges(source_node_ids, target_edge_ids)

        return self._get(selection, properties)

    def _iter_connections(self, source_node_ids, target_node_ids, unique_node_ids, shuffle):
        """ Iterate through `source_node_ids` -> `target_node_ids` connections. """
        # pylint: disable=too-many-branches,too-many-locals
        def _optimal_direction():
            """ Choose between source and target node IDs for iterating. """
            if target_node_ids is None and source_node_ids is None:
                raise BlueSnapError("Either `source` or `target` should be specified")
            if source_node_ids is None:
                return 'target'
            if target_node_ids is None:
                return 'source'
            else:
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
            primary_gids, secondary_gids = target_node_ids, source_node_ids
            get_connected_gids = self.afferent_nodes
        else:
            primary_gids, secondary_gids = source_node_ids, target_node_ids
            get_connected_gids = self.efferent_nodes

        primary_gids = np.unique(primary_gids)
        if shuffle:
            np.random.shuffle(primary_gids)

        if secondary_gids is not None:
            secondary_gids = np.unique(secondary_gids)

        secondary_gids_used = set()

        for key_gid in primary_gids:
            connected_gids = get_connected_gids(key_gid, unique=False)
            connected_gids_with_count = np.stack(
                np.unique(connected_gids, return_counts=True)
            ).transpose()
            # np.stack(uint64, int64) -> float64
            connected_gids_with_count = connected_gids_with_count.astype(np.uint32)
            if secondary_gids is not None:
                mask = np.in1d(connected_gids_with_count[:, 0], secondary_gids, assume_unique=True)
                connected_gids_with_count = connected_gids_with_count[mask]
            if shuffle:
                np.random.shuffle(connected_gids_with_count)
            for conn_gid, edge_count in connected_gids_with_count:
                if direction == 'target':
                    yield conn_gid, key_gid, edge_count
                else:
                    yield key_gid, conn_gid, edge_count
                if unique_node_ids:
                    secondary_gids_used.add(conn_gid)
                    break

    def iter_connections(
        self, source=None, target=None, unique_node_ids=False, shuffle=False,
        return_edge_ids=False, return_edge_count=False
    ):
        """
        Iterate through `source` -> `target` connections.

        Args:
            source: source node group
            target: target node group
            unique_node_ids: if True, no node ID would be used more than once
            shuffle: if True, result order would be (somewhat) randomised
            return_edge_count: if True, edge count is added to yield result
            return_edge_ids: if True, edge ID list is added to yield result

        `return_edge_count` and `return_edge_ids` are mutually exclusive.

        Yields:
            (source_node_id, target_node_id, edge_ids) if return_edge_ids == True;
            (source_node_id, target_node_id, edge_count) if return_edge_count == True;
            (source_node_id, target_node_id) otherwise.
        """
        if return_edge_ids and return_edge_count:
            raise BlueSnapError(
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
