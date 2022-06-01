import pytest
from kgforge.core import Resource
from mock import MagicMock, call

from bluepysnap.api import entity as test_module


def test_resolving_resource_simple():
    resource = Resource(id="id1", type="DetailedCircuit", name="fake_name")

    rr = test_module.ResolvingResource(resource)

    assert isinstance(rr, test_module.ResolvingResource)
    assert rr.name == "fake_name"
    assert rr.__repr__() == resource.__repr__()

    with pytest.raises(AttributeError, match="object has no attribute"):
        rr._non_existent

    with pytest.raises(AttributeError, match="object has no attribute"):
        rr.non_existent


def test_resolving_resource_nested():
    resource = Resource(
        id="id1",
        type="MorphologyRelease",
        morphologyIndex=Resource(id="id2", type="ModelReleaseIndex"),
    )
    # nested resource that will be returned by the mocked retriever
    nested = Resource(id="id2", type="ModelReleaseIndex", name="nested")
    nested._synchronized = True
    retriever = MagicMock(return_value=nested)

    rr = test_module.ResolvingResource(resource, retriever=retriever)

    assert isinstance(rr, test_module.ResolvingResource)
    assert rr.morphologyIndex.name == "nested"
    retriever.assert_called_once_with("id2")


def test_resolving_resource_metadata():
    resource = Resource(
        id="id1",
        type="MorphologyRelease",
    )
    resource._store_metadata = {"_meta_attr": "meta_value"}

    rr = test_module.ResolvingResource(resource)
    assert rr.meta_attr == "meta_value"


def test_entity():
    resource = Resource(
        id="id1", type="DetailedCircuit", name="fake_name", distribution="distribution"
    )

    def opener(res):
        return f"Instance of {res.id}"

    entity = test_module.Entity(resource, retriever=None, opener=opener)

    assert isinstance(entity, test_module.Entity)
    assert entity.resource == resource
    assert entity.instance == "Instance of id1"
    assert entity.id == "id1"
    with pytest.raises(AttributeError, match="object has no attribute"):
        entity.non_existent

    assert (
        entity.__repr__()
        == "Entity(resource_id=<id1>, resource_type=<DetailedCircuit>, instance_type=<Proxy>)"
    )

    entity = test_module.Entity(Resource())
    assert (
        entity.__repr__()
        == "Entity(resource_id=<None>, resource_type=<None>, instance_type=<NoneType>)"
    )


def test_entity_download():
    # Can't download
    downloader = MagicMock()
    resource = Resource(id="id1", type="DetailedCircuit", name="fake_name", distribution=None)
    entity = test_module.Entity(resource, downloader=downloader)
    entity.download()
    downloader.assert_not_called()

    # downloadable item in resource.distribution, default path
    resource = Resource(
        id="id1", type="DetailedCircuit", name="fake_name", distribution=["fake_item"]
    )
    entity = test_module.Entity(resource, downloader=downloader)
    entity.download()
    downloader.assert_called_once_with("fake_item", test_module.DOWNLOADED_CONTENT_PATH)

    # specify path and item to download
    downloader.reset_mock()
    entity.download(items=[1, 2], path="fake_path")
    downloader.has_calls((call(1, "fake_path"), call(2, "fake_path")))