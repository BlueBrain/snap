from kgforge.core import Resource
from mock import patch

from bluepysnap.api import core as test_module
from bluepysnap.api.entity import Entity
import pandas as pd


@patch(test_module.__name__ + ".KnowledgeGraphForge")
def test_api_init(mocked_forge):
    api = test_module.Api("fake_config.yml", bucket="fake/project", token="fake_token")
    assert isinstance(api, test_module.Api)


@patch(test_module.__name__ + ".KnowledgeGraphForge")
def test_api_get_entity_by_id(mocked_forge):
    mocked_forge.return_value.retrieve.return_value = Resource(id="id1", type="DetailedCircuit")
    api = test_module.Api("fake_config.yml", bucket="fake/project", token="fake_token")

    result = api.get_entity_by_id("id1")

    assert isinstance(result, Entity)
    assert result.id == "id1"
    assert result.type == "DetailedCircuit"


@patch(test_module.__name__ + ".KnowledgeGraphForge")
def test_api_get_entities_by_query(mocked_forge):
    mocked_forge.return_value.sparql.return_value = [Resource(id="id1")]
    mocked_forge.return_value.retrieve.return_value = Resource(id="id1", type="DetailedCircuit")
    api = test_module.Api("fake_config.yml", bucket="fake/project", token="fake_token")

    result = api.get_entities_by_query("FAKE QUERY")

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Entity)
    assert result[0].id == "id1"
    assert result[0].type == "DetailedCircuit"


@patch(test_module.__name__ + ".KnowledgeGraphForge")
def test_api_get_entities(mocked_forge):
    mocked_forge.return_value.sparql.return_value = [Resource(id="id1")]
    mocked_forge.return_value.retrieve.return_value = Resource(id="id1", type="DetailedCircuit")
    api = test_module.Api("fake_config.yml", bucket="fake/project", token="fake_token")

    result = api.get_entities("DetailedCircuit")

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Entity)
    assert result[0].id == "id1"
    assert result[0].type == "DetailedCircuit"


@patch(test_module.__name__ + ".KnowledgeGraphForge")
def test_api_as_dataframe(mocked_forge):
    # KnowledgeGraphForge.as_dataframe is patched so we can only mock the result
    df = pd.DataFrame(
        {
            "id": ["id1", "id2"],
            "type": ["DetailedCircuit", "DetailedCircuit"],
            "name": ["fake_name_1", "fake_name_2"],
        }
    )
    mocked_forge.return_value.as_dataframe.return_value = df
    api = test_module.Api("fake_config.yml", bucket="fake/project", token="fake_token")
    resources = [
        Resource(id="id1", type="DetailedCircuit", name="fake_name_1"),
        Resource(id="id2", type="DetailedCircuit", name="fake_name_2"),
    ]
    entities = [Entity(res) for res in resources]

    result = api.as_dataframe(entities)

    assert result is df
    mocked_forge.return_value.as_dataframe.assert_called_once_with(resources, store_metadata=True)


@patch(test_module.__name__ + ".KnowledgeGraphForge")
def test_api_as_json(mocked_forge):
    # KnowledgeGraphForge.as_json is patched so we can only mock the result
    resource_dict = {
        "id": "id1",
        "type": "DetailedCircuit",
        "name": "fake_name",
    }
    mocked_forge.return_value.as_json.return_value = resource_dict
    api = test_module.Api("fake_config.yml", bucket="fake/project", token="fake_token")
    resource = Resource(id="id1", type="DetailedCircuit", name="fake_name")
    entity = Entity(resource)

    result = api.as_json(entity)

    assert result is resource_dict
    mocked_forge.return_value.as_json.assert_called_once_with(resource, store_metadata=True)


@patch(test_module.__name__ + ".KnowledgeGraphForge")
def test_api_reopen(mocked_forge):
    api = test_module.Api("fake_config.yml", bucket="fake/project", token="fake_token")
    resource = Resource(id="id1", type="DetailedCircuit", name="fake_name")
    entity = Entity(resource)

    result = api.reopen(entity)
    assert isinstance(result, Entity)
    # only the wrapped resource is the same
    assert result.resource is entity.resource
    # while the instance is different
    assert result.instance is not entity.instance
