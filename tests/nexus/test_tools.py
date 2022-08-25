from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import bluepysnap.nexus.tools as test_module
from bluepysnap.nexus.entity import Entity


def test_wrap_morphology_dataframe_as_entities():
    df = pd.DataFrame(
        [
            ["fake_name_1", Path("/fake/path/1.fake")],
            ["fake_name_2", Path("/fake/path/2.fake")],
        ],
        columns=["name", "path"],
    )

    def reopen(*args, **_):
        return args[0]

    helper = MagicMock(reopen=reopen)
    res = test_module.wrap_morphology_dataframe_as_entities(df, helper)
    assert len(res) == len(df)
    for i, r in enumerate(res):
        assert isinstance(r, Entity)
        assert r.type == "NeuronMorphology"
        assert len(r.distribution) == 1
        assert r.distribution[0].name == df.iloc[i]["name"]
        assert r.distribution[0].type == "DataDownload"
        assert r.distribution[0].encodingFormat == "application/fake"
        assert r.distribution[0].atLocation.location == str(df.iloc[i]["path"])


def test_open_circuit_snap():
    class MockEntity:
        def __init__(self):
            self.circuitConfigPath = MagicMock(url="file:///fake/config/path")
            self.circuitBase = MagicMock(url="file:///fake/base/path")

    with patch("bluepysnap.Circuit") as patched:
        entity = MockEntity()

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
