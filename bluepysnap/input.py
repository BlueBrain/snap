# Copyright (c) 2020, EPFL/Blue Brain Project

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
"""Simulation input access."""
import libsonata

from bluepysnap.exceptions import BluepySnapError


class SynapseReplay:
    """Wrapper class for libsonata.SynapseReplay to provide the reader as a property."""

    def __init__(self, instance):
        """Wrap libsonata SynapseReplay object.

        Args:
            instance (libsonata.SynapseReplay): instance to wrap
        """
        self._instance = instance

    def __dir__(self):
        """Provide wrapped SynapseReplay instance's public attributes in dir."""
        public_attrs_instance = {attr for attr in dir(self._instance) if not attr.startswith("_")}
        return list(set(super().__dir__()) | public_attrs_instance)

    def __getattr__(self, name):
        """Retrieve attributes from the wrapped object."""
        return getattr(self._instance, name)

    @property
    def reader(self):
        """Return a spike reader object for the instance."""
        return libsonata.SpikeReader(self.spike_file)


def get_simulation_inputs(simulation):
    """Get simulation inputs as a dictionary.

    Args:
        simulation (libsonata.SimulationConfig): libsonata Simulation instance

    Returns:
        dict: inputs with input names as keys and corresponding objects as values
    """

    def _get_input(name):
        """Helper function to wrap certain objects."""
        item = simulation.input(name)

        if item.module.name == "synapse_replay":
            return SynapseReplay(item)
        return item

    if isinstance(simulation, libsonata.SimulationConfig):
        return {name: _get_input(name) for name in simulation.list_input_names}

    raise BluepySnapError(f"Unexpected type for 'simulation': {simulation.__class__.__name__}")
