import logging
import os
from unittest.mock import MagicMock

import pytest
from kgforge.core import KnowledgeGraphForge, Resource

from bluepysnap.nexus import connector as test_module


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
    forge.search.return_value = [resource]
    forge.retrieve.return_value = resource
    connector = test_module.NexusConnector(forge=forge)
    resource_type = "DetailedCircuit"
    resource_filter = {}

    result = connector.get_resources(resource_type, resource_filter, limit=1)

    assert result == [resource]
    forge.search.assert_called_once()
    forge.retrieve.assert_called_once()


def test_nexus_connector_download_resource(caplog):
    forge = MagicMock(KnowledgeGraphForge)
    forge.download.return_value = None
    connector = test_module.NexusConnector(forge=forge)

    class MockResource:
        def __init__(self, type_=""):
            self.name = ""
            self.contentUrl = ""
            self.type = type_

    with caplog.at_level(logging.WARNING):
        connector.download_resource(MockResource(), __file__)
        forge.download.assert_not_called()
        assert "already exists, not downloading..." in caplog.text

    connector.download_resource(MockResource("DataDownload"), "")
    forge.download.assert_called_once()

    with pytest.raises(RuntimeError, match="can not be downloaded"):
        connector.download_resource(MockResource("invalid_type"), "")

    with pytest.raises(RuntimeError, match="can not be downloaded"):
        resource = MockResource("DataDownload")
        del resource.contentUrl
        connector.download_resource(resource, "")


@pytest.mark.parametrize(
    "type_, filter_, expected",
    [
        (None, {}, {}),
        ("test_type", {}, {"type": "test_type"}),
        (None, {"test_filter": "test_value"}, {"test_filter": "test_value"}),
        (None, {"deprecated": True}, {"_deprecated": True}),
        (None, {"project": "test"}, {"_project": test_module.PROJECTS_NAMESPACE + "test"}),
        (None, {"createdBy": "test"}, {"_createdBy": test_module.USERS_NAMESPACE + "test"}),
        (None, {"updatedBy": "test"}, {"_updatedBy": test_module.USERS_NAMESPACE + "test"}),
        (None, {"createdAt": "test"}, {"_createdAt": "test" + test_module.DATETIME_SUFFIX}),
        (
            None,
            {"updatedBy": test_module.USERS_NAMESPACE + "test"},
            {"_updatedBy": test_module.USERS_NAMESPACE + "test"},
        ),
        (
            None,
            {"createdAt": "test" + test_module.DATETIME_SUFFIX},
            {"_createdAt": "test" + test_module.DATETIME_SUFFIX},
        ),
    ],
)
def test_search_builder(type_, filter_, expected):
    result = test_module._build_search_filters(type_, filter_)
    assert result == expected
