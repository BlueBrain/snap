from kgforge.core import Resource
from lazy_object_proxy import Proxy
from mock import MagicMock, patch

from bluepysnap.api import factory as test_module
from bluepysnap.api.entity import ResolvingResource


def test_entity_factory_init():
    factory = test_module.EntityFactory(connector=MagicMock())

    assert isinstance(factory, test_module.EntityFactory)


def test_entity_factory_get_registered_types():
    factory = test_module.EntityFactory(connector=MagicMock())

    result = factory.get_registered_types()

    assert result == {
        "DetailedCircuit",
        "Simulation",
        "MorphologyRelease",
        "DummyMorphology",
        "NeuronMorphology",
        "ReconstructedPatchedCell",
        "ReconstructedWholeBrainCell",
        "BrainAtlasRelease",
        "AtlasRelease",
    }


def test_entity_factory_get_available_tools():
    factory = test_module.EntityFactory(connector=MagicMock())
    tools_by_type = {
        "DetailedCircuit": ["snap", "bluepy"],
        "Simulation": ["snap", "bluepy", "bglibpy"],
        "MorphologyRelease": ["morph-tool"],
        "DummyMorphology": ["neurom"],
        "NeuronMorphology": ["neurom"],
        "ReconstructedPatchedCell": ["neurom"],
        "ReconstructedWholeBrainCell": ["neurom"],
        "BrainAtlasRelease": ["voxcell"],
        "AtlasRelease": ["voxcell"],
    }
    for resource_type, tools in tools_by_type.items():
        result = factory.get_available_tools(resource_type)

        assert (resource_type, result) == (resource_type, tools)


def test_entity_factory_open():
    factory = test_module.EntityFactory(connector=MagicMock())
    resource = Resource()

    result = factory.open(resource)

    assert result.wrapped is resource
    assert isinstance(result.resource, ResolvingResource)
    # no need to patch the opener because this will not resolve the proxy
    assert isinstance(result.instance, Proxy)


def test_entity_factory_open_instance():
    mocked_instance = MagicMock()
    with patch(
        test_module.__name__ + ".open_circuit_snap", return_value=mocked_instance
    ) as mocked_opener:
        factory = test_module.EntityFactory(connector=MagicMock())
        resource = Resource(type="DetailedCircuit")

        result = factory.open(resource, tool="snap")

        assert result.instance == mocked_instance
        mocked_opener.assert_called_once_with(result.resource)


def test_entity_factory_open_instance_with_default_tool():
    mocked_instance = MagicMock()
    with patch(
        test_module.__name__ + ".open_circuit_snap", return_value=mocked_instance
    ) as mocked_opener:
        factory = test_module.EntityFactory(connector=MagicMock())
        resource = Resource(type="DetailedCircuit")

        result = factory.open(resource)

        assert result.instance == mocked_instance
        mocked_opener.assert_called_once_with(result.resource)
