"""EdgePopulation stats helper."""

import numpy as np

from bluepysnap.exceptions import BluepySnapError


class StatsHelper:
    """EdgePopulation stats helper."""

    def __init__(self, edge_population):
        """Initialize StatsHelper with an EdgePopulation instance."""
        self._edge_population = edge_population

    def divergence(self, source, target, by, sample=None):
        """`source` -> `target` divergence.

        Calculate the divergence based on number of `"connections"` or `"synapses"` each `source`
        cell shares with the cells specified in `target`.
        * `connections`: number of unique target cells each source cell shares a connection with
        * `synapses`: number of unique synapses between a source cell and its target cells

        Args:
            source (int/CircuitNodeId/CircuitNodeIds/sequence/str/mapping/None): source nodes
            target (int/CircuitNodeId/CircuitNodeIds/sequence/str/mapping/None): target nodes
            by (str): 'synapses' or 'connections'
            sample (int): if specified, sample size for source group

        Returns:
            Array with synapse / connection count per each cell from `source` sample
            (taking into account only connections to cells in `target`).
        """
        by_alternatives = {"synapses", "connections"}
        if by not in by_alternatives:
            raise BluepySnapError(f"`by` should be one of {by_alternatives}; got: {by}")

        source_sample = self._edge_population.source.ids(source, sample=sample)

        result = {id_: 0 for id_ in source_sample}
        if by == "synapses":
            connections = self._edge_population.iter_connections(
                source_sample, target, return_synapse_count=True
            )
            for pre_gid, _, synapse_count in connections:
                result[pre_gid] += synapse_count
        else:
            connections = self._edge_population.iter_connections(source_sample, target)
            for pre_gid, _ in connections:
                result[pre_gid] += 1

        return np.array(list(result.values()))

    def convergence(self, source, target, by=None, sample=None):
        """`source` -> `target` convergence.

        Calculate the convergence based on number of `"connections"` or `"synapses"` each `target`
        cell shares with the cells specified in `source`.
        * `connections`: number of unique source cells each target cell shares a connection with
        * `synapses`: number of unique synapses between a target cell and its source cells

        Args:
            source (int/CircuitNodeId/CircuitNodeIds/sequence/str/mapping/None): source nodes
            target (int/CircuitNodeId/CircuitNodeIds/sequence/str/mapping/None): target nodes
            by (str): 'synapses' or 'connections'
            sample (int): if specified, sample size for target group

        Returns:
            Array with synapse / connection count per each cell from `target` sample
            (taking into account only connections from cells in `source`).
        """
        by_alternatives = {"synapses", "connections"}
        if by not in by_alternatives:
            raise BluepySnapError(f"`by` should be one of {by_alternatives}; got: {by}")

        target_sample = self._edge_population.target.ids(target, sample=sample)

        result = {id_: 0 for id_ in target_sample}
        if by == "synapses":
            connections = self._edge_population.iter_connections(
                source, target_sample, return_synapse_count=True
            )
            for _, post_gid, synapse_count in connections:
                result[post_gid] += synapse_count
        else:
            connections = self._edge_population.iter_connections(source, target_sample)
            for _, post_gid in connections:
                result[post_gid] += 1

        return np.array(list(result.values()))
