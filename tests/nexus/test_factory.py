import pytest
from kgforge.core import Resource
from lazy_object_proxy import Proxy
from mock import MagicMock, patch

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
        test_module.__name__ + ".open_circuit_snap", return_value=mocked_instance
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
        test_module.__name__ + ".open_circuit_snap", return_value=mocked_instance
    ) as mocked_opener:
        factory = test_module.EntityFactory(helper=MagicMock(), connector=MagicMock())
        resource = Resource(type="DetailedCircuit")

        result = factory.open(resource)

        assert isinstance(result.instance, Proxy)
        assert result.instance == mocked_instance
        mocked_opener.assert_called_once_with(result)

    with patch(
        test_module.__name__ + ".open_emodelconfiguration", return_value=mocked_instance
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
    with patch(
        test_module.__name__ + ".open_circuit_snap", side_effect=Exception("error")
    ) as mocked_opener:
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


def test_open_circuit_snap():
    class Entity:
        def __init__(self):
            self.circuitConfigPath = MagicMock(url="file:///fake/config/path")
            self.circuitBase = MagicMock(url="file:///fake/base/path")

    with patch("bluepysnap.Circuit") as patched:
        entity = Entity()

        test_module.open_circuit_snap(entity)
        patched.assert_called_once_with("/fake/config/path")
        patched.reset_mock()

        del entity.circuitConfigPath
        test_module.open_circuit_snap(entity)
        patched.assert_called_once_with("/fake/base/path/sonata/circuit_config.json")


def test_open_simulation_snap():
    with patch("bluepysnap.Simulation") as patched:
        entity = MagicMock()
        entity.path = "file:///fake/path"
        test_module.open_simulation_snap(entity)
        patched.assert_called_once_with("/fake/path/sonata/simulation_config.json")


def test_open_morphology_release():
    with patch("morph_tool.morphdb.MorphDB") as patched:
        entity = MagicMock()
        entity.morphologyIndex.distribution.url = "file:///fake/path"
        test_module.open_morphology_release(entity)
        patched.from_neurondb.assert_called_once_with("/fake/path")


def test__get_path_for_item():
    class MockItem:
        def __init__(self):
            self.name = "fake_name"
            self.contentUrl = MagicMock()
            self.atLocation = MagicMock(location=__file__)

    res = test_module._get_path_for_item(MockItem(), MagicMock())
    assert str(res) == __file__

    item = MockItem()
    entity = MagicMock()

    del item.atLocation
    res = test_module._get_path_for_item(item, entity)
    assert str(res) == str(test_module.DOWNLOADED_CONTENT_PATH / "fake_name")
    entity.download.assert_called_once_with(items=item, path=test_module.DOWNLOADED_CONTENT_PATH)

    del item.contentUrl
    res = test_module._get_path_for_item(item, entity)
    assert res is None


def test_open_morphology_neurom():
    entity = MagicMock()
    entity.distribution = [MagicMock(type="DataDownload", encodingFormat="application/swc")]
    with patch("neurom.io.utils.load_morphology") as neurom_patched:
        neurom_patched.return_value = "done"

        with patch(
            test_module.__name__ + "._get_path_for_item", MagicMock(return_value="/fake/path")
        ) as mock_get_path:
            result = test_module.open_morphology_neurom(entity)
            assert result == "done"
            mock_get_path.assert_called_once()

        neurom_patched.assert_called_once_with("/fake/path")

    entity.distribution = [MagicMock(type="fake_type")]

    with pytest.raises(RuntimeError, match="Missing morphology location"):
        test_module.open_morphology_neurom(entity)

    entity.distribution = [MagicMock(type="DataDownload", encodingFormat="unsupported/fmt")]

    with pytest.raises(RuntimeError, match="Unsupported morphology formats: {'unsupported/fmt'}"):
        test_module.open_morphology_neurom(entity)


def teusnthaoe(entity):
    import neurom

    # TODO: have a possibility to also read the file atLocation, if found and accessible?
    supported_formats = {"text/plain", "application/swc", "application/h5"}
    unsupported_formats = set()

    for item in always_iterable(entity.distribution):
        if item.type == "DataDownload":
            encoding_format = getattr(item, "encodingFormat", "").lower()
            if encoding_format in supported_formats:
                if hasattr(item, "atLocation"):
                    if hasattr(item.atLocation, "location"):
                        path = _get_path(item.atLocation.location)
                        if os.access(path, os.R_OK):
                            return neurom.io.utils.load_morphology(path)
                if hasattr(item, "contentUrl"):
                    entity.download(items=item, path=DOWNLOADED_CONTENT_PATH)
                    path = _get_downloaded_path(item.name)
                    return neurom.io.utils.load_morphology(path)
            if encoding_format:
                unsupported_formats.add(encoding_format)

    if unsupported_formats:
        raise RuntimeError(f"Unsupported morphology formats: {unsupported_formats}")

    raise RuntimeError("Missing morphology location")
