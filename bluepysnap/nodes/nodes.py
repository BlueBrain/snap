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

"""Nodes access."""

import numpy as np

from bluepysnap._doctools import AbstractDocSubstitutionMeta
from bluepysnap.circuit_ids import CircuitNodeIds
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.network import NetworkObject
from bluepysnap.nodes.node_population import NodePopulation


class Nodes(
    NetworkObject,
    metaclass=AbstractDocSubstitutionMeta,
    source_word="NetworkObject",
    target_word="Node",
):
    """The top level Nodes accessor."""

    _population_class = NodePopulation

    def __init__(self, circuit):  # pylint: disable=useless-super-delegation
        """Initialize the top level Nodes accessor."""
        super().__init__(circuit)

    @property
    def population_names(self):
        """Defines all sorted node population names from the Circuit."""
        return sorted(self._circuit.to_libsonata.node_populations)

    def property_values(self, prop):
        """Returns all the values for a given Nodes property."""
        return set(
            value
            for pop in self.values()
            if prop in pop.property_names
            for value in pop.property_values(prop)
        )

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
                raise BluepySnapError(f"Population {diff} does not exist in the circuit.")

        fun = lambda x: (x.ids(group, raise_missing_property=False), x.name)
        return self._get_ids_from_pop(fun, CircuitNodeIds, sample=sample, limit=limit)

    def get(self, group=None, properties=None):  # pylint: disable=arguments-differ
        """Node properties by iterating populations.

        Args:
            group (CircuitNodeIds/int/sequence/str/mapping/None): Which nodes will have their
                properties returned depends on the type of the ``group`` argument:
                See :py:class:`~bluepysnap.nodes.Nodes.ids`.

            properties (str/list): If specified, return only the properties in the list.
                Otherwise return all properties.

        Returns:
            generator: yields tuples of ``(<population_name>, pandas.DataFrame)``:
                - DataFrame indexed by CircuitNodeIds containing the properties from ``properties``.

        Notes:
            The NodePopulation.property_names function will give you all the usable properties
            for the `properties` argument.
        """
        if properties is None:
            # not strictly needed, but ensure that the properties are always in the same order
            properties = sorted(self.property_names)
        return super().get(group, properties)

    def __getstate__(self):
        """Make Nodes pickle-able, without storing state of caches."""
        return self._circuit

    def __setstate__(self, state):
        """Load from pickle state."""
        self.__init__(state)
