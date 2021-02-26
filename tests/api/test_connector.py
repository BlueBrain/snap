import pytest
from kgforge.core import KnowledgeGraphForge, Resource
from mock import MagicMock

from bluepysnap.api import connector as test_module


def assert_query_equals(actual, expected):
    def _clean(s):
        return " ".join(s.split())

    assert _clean(actual) == _clean(expected)


def test_nexus_connector_init():
    forge = MagicMock()
    connector = test_module.NexusConnector(forge=forge)

    assert isinstance(connector, test_module.NexusConnector)


def test_nexus_connector_query():
    resources = [Resource(id="id1"), Resource(id="id2")]
    forge = MagicMock(KnowledgeGraphForge)
    forge.sparql.return_value = resources
    connector = test_module.NexusConnector(forge=forge)

    result = connector.query("FAKE QUERY")

    assert result == resources
    forge.sparql.assert_called_once()


def test_nexus_connector_get_resource_by_id():
    resource = Resource(id="id1")
    forge = MagicMock(KnowledgeGraphForge)
    forge.retrieve.return_value = resource
    connector = test_module.NexusConnector(forge=forge)

    result = connector.get_resource_by_id("id1")

    assert result == resource
    forge.retrieve.assert_called_once()


def test_nexus_connector_get_resources_by_query():
    resource = Resource(id="id1")
    forge = MagicMock(KnowledgeGraphForge)
    forge.sparql.return_value = [resource]
    forge.retrieve.return_value = resource
    connector = test_module.NexusConnector(forge=forge)

    result = connector.get_resources_by_query("FAKE QUERY")

    assert result == [resource]
    forge.sparql.assert_called_once()
    forge.retrieve.assert_called_once()


def test_nexus_connector_get_resources():
    resource = Resource(id="id1")
    forge = MagicMock(KnowledgeGraphForge)
    forge.sparql.return_value = [resource]
    forge.retrieve.return_value = resource
    connector = test_module.NexusConnector(forge=forge)
    resource_type = "DetailedCircuit"
    resource_filter = {}  # other filters are tested in QueryBuilder tests

    result = connector.get_resources(resource_type, resource_filter, limit=1)

    assert result == [resource]


@pytest.mark.parametrize(
    "resource_type, resource_filter, expected",
    [
        (
            "DetailedCircuit",
            (),
            """
            PREFIX nexus: <https://bluebrain.github.io/nexus/vocabulary/>
            SELECT ?id ?project
            WHERE {
                ?id
                a DetailedCircuit ;
                nexus:project ?project ;
                nexus:createdAt ?createdAt ;
                nexus:deprecated false .
            }
            ORDER BY DESC(?createdAt)
            """,
        ),
        (
            "DetailedCircuit",
            {
                "project": "fake_project",
                "createdBy": "fake_user",
            },
            """
            PREFIX nexus: <https://bluebrain.github.io/nexus/vocabulary/>
            SELECT ?id ?project
            WHERE {
                ?id
                a DetailedCircuit ;
                nexus:project ?project ;
                nexus:createdAt ?createdAt ;
                nexus:deprecated false ;
                nexus:project <https://bbp.epfl.ch/nexus/v1/projects/fake_project> ;
                nexus:createdBy <https://bbp.epfl.ch/nexus/v1/realms/bbp/users/fake_user> .
            }
            ORDER BY DESC(?createdAt)
            """,
        ),
        (
            "DetailedCircuit",
            {
                "createdBy": ["fake_user_1", "fake_user_2"],
            },
            """
            PREFIX nexus: <https://bluebrain.github.io/nexus/vocabulary/>
            SELECT ?id ?project
            WHERE {
                ?id
                a DetailedCircuit ;
                nexus:project ?project ;
                nexus:createdAt ?createdAt ;
                nexus:deprecated false ;
                nexus:createdBy ?o0 .
            FILTER(?o0 IN
                (<https://bbp.epfl.ch/nexus/v1/realms/bbp/users/fake_user_1>,
                <https://bbp.epfl.ch/nexus/v1/realms/bbp/users/fake_user_2>))
            }
            ORDER BY DESC(?createdAt)
            """,
        ),
        (
            "DetailedCircuit",
            {
                "brainLocation.brainRegion.label": "fake_region",
            },
            """
            PREFIX nexus: <https://bluebrain.github.io/nexus/vocabulary/>
            SELECT ?id ?project
            WHERE {
                ?id
                a DetailedCircuit ;
                nexus:project ?project ;
                nexus:createdAt ?createdAt ;
                nexus:deprecated false ;
                brainLocation / brainRegion / label 'fake_region' .
            }
            ORDER BY DESC(?createdAt)
            """,
        ),
        (
            "DetailedCircuit",
            {
                "nodeCollection.memodelRelease.morphologyRelease.name": "fake_morph_release",
            },
            """
            PREFIX nexus: <https://bluebrain.github.io/nexus/vocabulary/>
            SELECT ?id ?project
            WHERE {
                ?id
                a DetailedCircuit ;
                nexus:project ?project ;
                nexus:createdAt ?createdAt ;
                nexus:deprecated false ;
                nodeCollection / memodelRelease / morphologyRelease / name 'fake_morph_release' .
            }
            ORDER BY DESC(?createdAt)
            """,
        ),
        (
            "MorphologyRelease",
            {
                "^morphologyRelease.^memodelRelease.^nodeCollection.name": "fake_circuit",
            },
            """
            PREFIX nexus: <https://bluebrain.github.io/nexus/vocabulary/>
            SELECT ?id ?project
            WHERE {
                ?id
                a MorphologyRelease ;
                nexus:project ?project ;
                nexus:createdAt ?createdAt ;
                nexus:deprecated false ;
                ^morphologyRelease / ^memodelRelease / ^nodeCollection / name 'fake_circuit' .
            }
            ORDER BY DESC(?createdAt)
            """,
        ),
        (
            "SimulationCampaign",
            {
                "used": "<https://bbp.epfl.ch/nexus/v1/resources/nse/test/_/fake_id>",
            },
            """
            PREFIX nexus: <https://bluebrain.github.io/nexus/vocabulary/>
            SELECT ?id ?project
            WHERE {
                ?id
                a SimulationCampaign ;
                nexus:project ?project ;
                nexus:createdAt ?createdAt ;
                nexus:deprecated false ;
                used <https://bbp.epfl.ch/nexus/v1/resources/nse/test/_/fake_id> .
            }
            ORDER BY DESC(?createdAt)
            """,
        ),
    ],
)
def test_query_builder_build_query(resource_type, resource_filter, expected):
    qb = test_module.QueryBuilder()
    result = qb.build_query(resource_type, resource_filter)
    assert_query_equals(result, expected)
