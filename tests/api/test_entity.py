from kgforge.core import Resource
from mock import MagicMock

from bluepysnap.api import entity as test_module


def test_resolving_resource_simple():
    resource = Resource(id="id1", type="DetailedCircuit", name="fake_name")

    rr = test_module.ResolvingResource(resource)

    assert isinstance(rr, test_module.ResolvingResource)
    assert rr.name == "fake_name"


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


def test_entity():
    resource = Resource(id="id1", type="DetailedCircuit", name="fake_name")
    rr = test_module.ResolvingResource(resource)
    instance = MagicMock()

    entity = test_module.Entity(resource=rr, instance=instance)

    assert isinstance(entity, test_module.Entity)
    assert entity.resource == rr
    assert entity.wrapped == resource
    assert entity.instance == instance
    assert entity.id == resource.id
