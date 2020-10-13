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
import logging
from cached_property import cached_property

# from bluepysnap import Circuit, Simulation
from bluepysnap import Circuit
from bluepy.v2 import Simulation as BluepySimulation, Circuit as BluepyCircuit

L = logging.getLogger(__name__)


def _statement_deprecated(deprecated):
    """Helper function to return a query statement for deprecated property."""
    keyword = 'https://bluebrain.github.io/nexus/vocabulary/deprecated'
    bool_str = str(bool(deprecated)).lower()
    return '?id <{}> {} .'.format(keyword, bool_str)


def _statement_creator(username):
    """Helper function to return a query statement for the creator of the property."""
    keyword = 'https://bluebrain.github.io/nexus/vocabulary/createdBy'
    return '?id <{}> ?name . FILTER regex(str(?name), "{}") .'.format(keyword, username)


def _statement_created_at():
    return '?id <https://bluebrain.github.io/nexus/vocabulary/createdAt> ?date .'


def _statement_species(species):
    """Helper function to return a query statement for the species of the property."""
    statement = '?id species ?species . ?species label ?l . FILTER (?l IN ("{}")) .'
    return statement.format(species)


def _statement_type(type_):
    return '?id a {} .'.format(type_)


statements_map = {
    'creator': _statement_creator,
    'species': _statement_species,
    'type': _statement_type,
    'deprecated': _statement_deprecated}


def query_resources(forge, filters):
    """Perform a query with filters defined in a dict"""
    query = 'SELECT ?id WHERE {{{}}}'
    statements = []
    for key, value in filters.items():
        if key in statements_map:
            statements.append(statements_map[key](value))

    statements.append(_statement_created_at())
    query = query.format(' '.join(statements))
    limit = filters.get('limit')

    if limit:
        query = '{} ORDER BY DESC(?date)'.format(query)

    return forge.sparql(query, limit=limit)


class KGResource(object):
    """Base class for Knowledge Graph resources."""

    def __init__(self, resource, forge):
        """Wrapper for KnowledgeGraphForge Resource.

        Args:
            resource (kgforge.core.Resource): kgforge Resource instance.
            forge (KnowledgeGraphForge): KnowledgeGraphForge instance used to access the data.

        Returns:
            KGResource: A KGResource object.
        """
        self._kg_resource = resource
        self._forge = forge

    @classmethod
    def from_id(cls, id_, forge):
        """Initializes a Knowledge Graph resource using its id (URL).

        Args:
            id_ (str): id  of the resource in Knowledge Graph.
            forge (KnowledgeGraphForge): KnowledgeGraphForge instance used to access the data.
        """
        return cls(forge.retrieve(id_, cross_bucket=True), forge)

    @classmethod
    def from_query(cls, resources, forge):
        """TODO: Docstring for from_query.

        :arg1: TODO
        :returns: TODO

        """
        return [cls.from_id(resource.id, forge) for resource in resources]

    @property
    def metadata(self):
        """Metadata of the Knowledge Graph resource."""
        return self._kg_resource

    @property
    def store_metadata(self):
        """Metadata of the Knowledge Graph store."""
        # pylint: disable=W0212
        # Access to store metadata can only be done by accessing a protected member.
        return self.metadata._store_metadata

    @property
    def creator(self):
        """The creator of the resource as described in Knowledge Graph."""
        return self.store_metadata.get('_createdBy')

    @property
    def created_at(self):
        """The creation time of the resource as described in Knowledge Graph."""
        return self.store_metadata.get('_createdAt')

    @property
    def forge(self):
        """The KnowledgeGraphForge instance."""
        return self._forge

    @property
    def project(self):
        """The project name as described in Knowledge Graph."""
        return self.store_metadata.get('_project')

    def _query(self, query_str):
        """Make a query using the KnowledgeGraphForge's sparql function."""
        return self.forge.sparql(query_str)

    def __getattr__(self, attribute):
        """To ease the access to metadata.

        E.g., using resource.name instead of resource.metadata.name.
        """
        if hasattr(self.metadata, attribute):
            attribute = getattr(self.metadata, attribute)
            if hasattr(attribute, 'id'):
                attribute = KGResource.from_id(getattr(attribute, 'id'), self.forge)

            return attribute

        return self.__getattribute__(attribute)


# class KGCircuit(KGResource, Circuit):
class KGCircuit(KGResource):
    """Access to circuit data in Knowledge Graph."""

    def __init__(self, resource, forge):
        """Initializes a Knowledge Graph Circuit resource using its url.

        Args:
            resource (kgforge.core.Resource): kgforge Resource instance.
            forge (KnowledgeGraphForge): KnowledgeGraphForge instance used to access the data.

        Returns:
            KGResource: A KGCircuit object.
        """
        KGResource.__init__(self, resource, forge)
        # Circuit.__init__(self, self.config_path)

    @cached_property
    def simulation_campaigns(self):
        """Retrieve all simulation campaigns that use this circuit."""
        resources = self._query('SELECT ?id WHERE {{\
            ?id a SimulationCampaign ; used <{}> . }}'.format(self.id))

        return KGSimulationCampaign.from_query(resources, self.forge)

    @cached_property
    def simulations(self):
        """Retrieve all simulations that use this circuit either directly or in a campaign."""
        resources = self._query(
            'SELECT DISTINCT ?id WHERE {{\
                {{ ?id a Simulation ; wasStartedBy ?scid .\
                   ?scid a SimulationCampaign ; used <{}> . }}\
            UNION\
                {{ ?id a Simulation ; used <{}> . }}\
            }}'.format(self.id, self.id))

        return KGSimulation.from_query(resources, self.forge)

    @cached_property
    def morphology_release(self):
        """Retrieve the morphology release for the circuit."""
        # resources = self._query('SELECT ?id WHERE {{\
        #     <{}> a DetailedCircuit; nodeCollection ?nid .\
        #     ?nid a NodeCollection; memodelRelease ?mid .\
        #     ?mid a MEModelRelease ; morphologyRelease ?id . }}'.format(self.id))

        resources = self._query('SELECT ?id WHERE {{\
            <{}> (<>|!<>)* ?id .\
            ?id a MorphologyRelease . }}'.format(self.id))

        return KGResource.from_query(resources, self.forge)[0]

    @cached_property
    def config_path(self):
        """The path to SONATA config file."""
        base_path = self.metadata.circuitBase.url.replace('file://', '')
        return os.path.join(base_path, 'sonata/circuit_config.json')


# class KGSimulation(KGResource, Simulation):
class KGSimulation(KGResource, BluepySimulation):
    """Access to simulation data in Knowledge Graph."""

    def __init__(self, resource, forge):
        """Initializes a Knowledge Graph Simulation resource using its url.

        Args:
            resource (kgforge.core.Resource): kgforge Resource instance.
            forge (KnowledgeGraphForge): KnowledgeGraphForge instance used to access the data.

        Returns:
            KGResource: A KGSimulation object.
        """
        KGResource.__init__(self, resource, forge)

        if hasattr(self, 'path'):
            BluepySimulation.__init__(self, self.config_path_bluepy)
            # Simulation.__init__(self, self.config_path)
        else:
            L.warn('No config path available for simulation')

    @cached_property
    def config_path(self):
        """The path to SONATA config file."""
        return os.path.join(self.path, 'sonata/simulation_config.json')

    @property
    def config_path_bluepy(self):
        """The path to BluePy config file."""
        return os.path.join(self.path, 'BlueConfig')

    @cached_property
    def simulation_campaign(self):
        """The simulation campaign that launched the simulation."""
        return KGSimulationCampaign.from_id(self.metadata.wasStartedBy.id, self.forge)

    # Commented out due to having to use bluepy for demoing purposes.
    # @property
    # def circuit(self):
        # """Access to the circuit used for the simulation."""
        # return self.simulation_campaign.circuit


class KGCircuitSonata(KGCircuit, Circuit):
    """KGCircuit class that inherits Circuit from sonata."""

    def __init__(self, resource, forge):
        KGCircuit.__init__(self, resource, forge)
        Circuit.__init__(self, self.config_path)


class KGCircuitBluepy(KGCircuit, BluepyCircuit):
    """KGCircuit class that inherits Circuit from bluepy."""

    def __init__(self, resource, forge):
        KGCircuit.__init__(self, resource, forge)
        BluepyCircuit.__init__(self, self.config_path)

    @cached_property
    def config_path(self):
        """The path to Bluepy config file."""
        base_path = self.metadata.circuitBase.url.replace('file://', '')
        return os.path.join(base_path, 'CircuitConfig')


class KGSimulationCampaign(KGResource):
    """Access to simulation campaign data in Knowledge Graph."""

    @cached_property
    def simulations(self):
        """Get all the simulations of this simulation campaign."""
        resources = self._query('SELECT DISTINCT ?id WHERE {{\
            ?id a Simulation ; wasStartedBy <{}> . }}'.format(self.id))

        return KGSimulation.from_query(resources, self.forge)

    @cached_property
    def circuit(self):
        """Access to the circuit used for the simulation campaign."""
        circuit_id = next((u.id for u in self.used if u.type == 'DetailedCircuit'), None)

        return KGCircuit.from_id(circuit_id, self.forge)
