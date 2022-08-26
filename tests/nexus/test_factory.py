from unittest.mock import MagicMock, patch

import pytest
from kgforge.core import Resource
from lazy_object_proxy import Proxy

from bluepysnap.nexus import factory as test_module


def test_entity_factory_init():
    factory = test_module.EntityFactory(helper=MagicMock(), connector=MagicMock())

    assert isinstance(factory, test_module.EntityFactory)


def test_entity_factory_get_registered_types():
    factory = test_module.EntityFactory(helper=MagicMock(), connector=MagicMock())

    result = factory.get_registered_types()

    assert result == {
        "DetailedCircuit",
        "Simulation",
        "MorphologyRelease",
        "NeuronMorphology",
        "ReconstructedCell",
        "ReconstructedPatchedCell",
        "ReconstructedWholeBrainCell",
        "BrainAtlasRelease",
        "AtlasRelease",
        "EModelConfiguration",
    }


def test_entity_factory_get_available_tools():
    factory = test_module.EntityFactory(helper=MagicMock(), connector=MagicMock())
    tools_by_type = {
        "DetailedCircuit": ["snap", "bluepy"],
        "Simulation": ["snap", "bluepy", "bglibpy"],
        "MorphologyRelease": ["morph-tool"],
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
    factory = test_module.EntityFactory(helper=MagicMock(), connector=MagicMock())
    resource = Resource()

    result = factory.open(resource)

    assert result.resource is resource
    # no need to patch the opener in this test,
    # because the proxy is not resolved when calling `isinstance`
    assert isinstance(result.instance, Proxy)


def test_entity_factory_open_instance():
    mocked_instance = MagicMock()
    with patch(
        "bluepysnap.nexus.tools.open_circuit_snap", return_value=mocked_instance
    ) as mocked_opener:
        factory = test_module.EntityFactory(helper=MagicMock(), connector=MagicMock())
        resource = Resource(type="DetailedCircuit")

        result = factory.open(resource, tool="snap")

        assert isinstance(result.instance, Proxy)
        assert result.instance == mocked_instance
        mocked_opener.assert_called_once_with(result)


def test_entity_factory_open_instance_with_default_tool():
    mocked_instance = MagicMock()
    with patch(
        "bluepysnap.nexus.tools.open_circuit_snap", return_value=mocked_instance
    ) as mocked_opener:
        factory = test_module.EntityFactory(helper=MagicMock(), connector=MagicMock())
        resource = Resource(type="DetailedCircuit")

        result = factory.open(resource)

        assert isinstance(result.instance, Proxy)
        assert result.instance == mocked_instance
        mocked_opener.assert_called_once_with(result)

    with patch(
        "bluepysnap.nexus.tools.open_emodelconfiguration", return_value=mocked_instance
    ) as mocked_opener:
        connector = MagicMock()
        resource = Resource(type="EModelConfiguration")
        factory = test_module.EntityFactory(helper=MagicMock(), connector=connector)
        result = factory.open(resource)
        assert result.instance == mocked_instance
        mocked_opener.assert_called_once_with(result, connector)


def test_entity_factory_open_instance_with_non_default_tool():
    mocked_instance = MagicMock()
    factory = test_module.EntityFactory(helper=MagicMock(), connector=MagicMock())
    tool = MagicMock()
    tool.return_value = mocked_instance

    factory.register("TestType", "test_tool", tool)
    resource = Resource(type="TestType")

    result = factory.open(resource)
    assert isinstance(result.instance, Proxy)
    assert result.instance == mocked_instance
    tool.assert_called_once_with(result)

    tool.reset_mock()
    with pytest.raises(RuntimeError, match="Tool .* not found for"):
        result = factory.open(resource, tool="fake_tool")
        print(result.instance)


def test_entity_factory_fail_to_open_instance():
    with patch("bluepysnap.nexus.tools.open_circuit_snap", side_effect=Exception("error")):
        factory = test_module.EntityFactory(helper=MagicMock(), connector=MagicMock())
        resource = Resource(type="DetailedCircuit")

        with pytest.raises(RuntimeError, match="Unable to open"):
            result = factory.open(resource)

            assert isinstance(result.instance, Proxy)
            print(result.instance)


def test__get_tool_functions_errors():
    factory = test_module.EntityFactory(helper=MagicMock(), connector=MagicMock())
    with pytest.raises(RuntimeError, match="No available tools to open"):
        factory._get_tool_functions(("fake_type"))

    with pytest.raises(RuntimeError, match="Multiple tools to open"):
        factory._get_tool_functions(("DetailedCircuit", "EModelConfiguration"))
