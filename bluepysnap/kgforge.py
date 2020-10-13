"""KnowledgeGraphForge wrappings for SNAP.

General notes:
    User logging in runtime not yet implemented
    - Did some tests, and there seems to be an issue regarding the Oauth2
      client secret which we do not have.
    - Should be investigated further.
    - So for now, need to copy the token for the kgforge

    Getting resources created by current user not implemented as of yet
    - Would be simple, if we had a Oauth2 login and could get the username from there

    Simulation not tested yet due to there not being Simulations with SONATA config in Nexus
    - SNAP does not support BlueConfig

    Currently the functions retrieve all the instances in the KG, also the deprecated ones
    - if this is not wanted, there is a way to exclude them in the sparql queries

    KnowledgeGraphForge instance is passed to the classes' __init__ function
    - I would prefer a global instance of it but then it would need to be changed if accessing
      data in other 'buckets' (org/project)
    - not sure what is the best approach for this or if this is going to be simpler in the future.
"""


import os
from cached_property import cached_property

from kgforge.specializations.stores.nexus.service import DEPRECATED_PROPERTY

from bluepysnap import Circuit, Simulation


def _statement_deprecated(deprecated):
    """Helper function to return a query statement for deprecated property."""
    bool_str = 'true' if deprecated else 'false'
    return '<{}> {}'.format(DEPRECATED_PROPERTY, bool_str)


class KGResource(object):
    """Base class for Knowledge Graph resources."""
    # For now, access to store metadata (creator, etc.) can only be done by accessing a protected
    # member.
    # pylint: disable=W0212

    def __init__(self, url, forge):
        """Initializes a Knowledge Graph resource using its url.

        Args:
            url (str): URL to the resource in Knowledge Graph.
            forge (KnowledgeGraphForge): KnowledgeGraphForge instance used to access the data.

        Returns:
            KGResource: A KGResource object.
        """
        self._kg_resource = forge.retrieve(url)
        self._forge = forge

    @property
    def metadata(self):
        """Metadata of the Knowledge Graph resource."""
        return self._kg_resource

    @property
    def creator(self):
        """The creator of the resource as described in Knowledge Graph."""
        return self.metadata._store_metadata.get('_createdBy')

    @property
    def created_at(self):
        """The creation time of the resource as described in Knowledge Graph."""
        return self.metadata._store_metadata.get('_createdAt')

    @property
    def forge(self):
        """The KnowledgeGraphForge instance."""
        return self._forge

    @property
    def project(self):
        """The project name as described in Knowledge Graph."""
        return self.metadata._store_metadata.get('_project')

    def _query_retrieve(self, query_str):
        """Make a query and retrieve each of the resulted resources.

        NOTE: need to be looked into if there is a more sophisticated and more effective
        way of doing this. With a lot of resources this takes time. If they are not 'retrieved',
        not much useful info is available.
        """
        return [self.forge.retrieve(item.id) for item in self.forge.sparql(query_str)]

    def __getattr__(self, attribute):
        """To ease the access to metadata.

        E.g., using resource.name instead of resource.metadata.name.
        """
        if hasattr(self.metadata, attribute):
            return getattr(self.metadata, attribute)

        return self.__getattribute__(attribute)


class KGCircuit(KGResource, Circuit):
    """Access to circuit data in Knowledge Graph."""

    def __init__(self, url, forge):
        """Initializes a Knowledge Graph Circuit resource using its url.

        Args:
            url (str): URL to the resource in Knowledge Graph.
            forge (KnowledgeGraphForge): KnowledgeGraphForge instance used to access the data.

        Returns:
            KGResource: A KGCircuit object.
        """
        KGResource.__init__(self, url, forge)
        Circuit.__init__(self, self.config_path)

    @cached_property
    def simulation_campaigns(self):
        """Retrieve all simulation campaigns that use this circuit."""
        return self._query_retrieve('SELECT ?id WHERE {{\
            ?id a SimulationCampaign ; used <{}> . }}'.format(self.id))

    @cached_property
    def simulations(self):
        """Retrieve all simulations that use this circuit."""
        return self._query_retrieve('SELECT DISTINCT ?id WHERE {{\
            ?id a Simulation ; wasStartedBy ?scid .\
            ?scid a SimulationCampaign ; used <{}> .  }}'.format(self.id))

    @cached_property
    def config_path(self):
        """The path to SONATA config file."""
        base_path = self.metadata.circuitBase.url.replace('file://', '')
        return os.path.join(base_path, 'sonata/circuit_config.json')


class KGSimulation(KGResource, Simulation):
    """Access to simulation data in Knowledge Graph."""

    def __init__(self, url, forge):
        """Initializes a Knowledge Graph Simulation resource using its url.

        Args:
            url (str): URL to the resource in Knowledge Graph.
            forge (KnowledgeGraphForge): KnowledgeGraphForge instance used to access the data.

        Returns:
            KGResource: A KGSimulation object.
        """
        KGResource.__init__(self, url, forge)
        Simulation.__init__(self, self.config_path)

    @cached_property
    def config_path(self):
        """The path to SONATA config file."""
        return os.path.join(self.path, 'sonata/simulation_config.json')

    @cached_property
    def simulation_campaign(self):
        """The simulation campaign that launched the simulation."""
        return self.forge.retrieve(self.metadata.wasStartedBy.id)

    @cached_property
    def circuit(self):
        """Access to the circuit used for the simulation."""
        circuit_id = next((u for u in self.simulation_campaign.used if u.type == 'DetailedCircuit'),
                          None)
        return KGCircuit(circuit_id, self.forge)


class KGSimulationCampaign(KGResource):
    """Access to simulation campaign data in Knowledge Graph."""

    def __init__(self, url, forge):
        """Initializes a Knowledge Graph Simulation resource using its url.

        Args:
            url (str): URL to the resource in Knowledge Graph.
            forge (KnowledgeGraphForge): KnowledgeGraphForge instance used to access the data.

        Returns:
            KGResource: A KGSimulation object.
        """
        KGResource.__init__(self, url, forge)

    @cached_property
    def simulations(self):
        """Retrieve all simulations in this simulation campaign."""
        return self._query_retrieve('SELECT DISTINCT ?id WHERE {{\
            ?id a Simulation ; wasStartedBy <{}> . }}'.format(self.id))
