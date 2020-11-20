import pytest
from pathlib2 import Path
import json

import bluepysnap.config as test_module
from bluepysnap.exceptions import BluepySnapError

from utils import TEST_DATA_DIR, edit_config, copy_config


def test_parse():
    actual = test_module.Config.parse(
        str(TEST_DATA_DIR / 'circuit_config.json')
    )

    # check double resolution and '.' works: $COMPONENT_DIR -> $BASE_DIR -> '.'
    assert actual['components']['morphologies_dir'] == str(TEST_DATA_DIR / 'morphologies')

    # check resolution and './' works: $NETWORK_DIR -> './'
    assert actual['networks']['nodes'][0]['nodes_file'] == str(TEST_DATA_DIR / 'nodes.h5')

    # check resolution of '../' works: $PARENT --> '../'
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["manifest"]["$PARENT"] = "../"
            config["components"]["other"] = "$PARENT/other"

        actual = test_module.Config.parse(config_path)
        assert (
                actual['components']['other'] ==
                str(Path(config_path.parent / "../other").resolve())
        )

    # check resolution of '../' works in a path outside manifest
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["components"]["other"] = "../other"

        actual = test_module.Config.parse(config_path)
        assert (
                actual['components']['other'] ==
                str(Path(config_path.parent / "../other").resolve())
        )

    # check resolution without manifest of '../' works in a path outside
    # i.e. : self.manifest contains the configdir even if manifest is not here
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            for k in list(config): config.pop(k)
            config["something"] = "../other"
        actual = test_module.Config.parse(config_path)
        assert actual["something"] == str(Path(config_path.parent / "../other").resolve())

    # check resolution with multiple slashes
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["something"] = "$COMPONENT_DIR/something////else"
        actual = test_module.Config.parse(config_path)
        assert actual["something"] == str(Path(config_path.parent) / 'something' / 'else')

    # check resolution with $ in a middle of the words
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["something"] = "$COMPONENT_DIR/somet$hing/else"
        actual = test_module.Config.parse(config_path)
        assert actual["something"] == str(Path(config_path.parent) / 'somet$hing' / 'else')

    # check resolution with relative path without "." in the manifest
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["manifest"]["$NOPOINT"] = "nopoint"
            config["components"]["other"] = "$NOPOINT/other"
        actual = test_module.Config.parse(config_path)
        assert actual["components"]["other"] == str(Path(config_path.parent) / 'nopoint' / 'other')

    # check resolution for non path objects
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            for k in list(config): config.pop(k)
            config["string"] = "string"
            config["int"] = 1
            config["double"] = 0.2
            # just to check because we use starting with '.' as a special case
            config["tricky_double"] = .2
            config["path"] = "./path"

        actual = test_module.Config.parse(config_path)

        # string
        assert actual["string"] == "string"
        # int
        assert actual["int"] == 1
        # double
        assert actual["double"] == 0.2
        assert actual["tricky_double"] == 0.2
        # path
        assert actual["path"] == str(Path(config_path.parent / "./path").resolve())


def test_bad_manifest():
    # 2 anchors would result in the absolute path of the last one : misleading
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["manifest"]["$COMPONENT_DIR"] = "$BASE_DIR/$NETWORK_DIR"
        with pytest.raises(BluepySnapError):
            test_module.Config.parse(config_path)

    # same but not in the manifest
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["components"]["other"] = "$COMPONENT_DIR/$BASE_DIR"
        with pytest.raises(BluepySnapError):
            test_module.Config.parse(config_path)

    # relative path with an anchor in the middle is not allowed this breaks the purpose of the
    # anchors (they are not just generic placeholders)
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["components"]["other"] = "something/$COMPONENT_DIR/"
        with pytest.raises(BluepySnapError):
            test_module.Config.parse(config_path)

    # abs path with an anchor in the middle is not allowed
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["components"]["other"] = "/something/$COMPONENT_DIR/"
        with pytest.raises(BluepySnapError):
            test_module.Config.parse(config_path)

    # unknown anchor
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["components"]["other"] = "$UNKNOWN/something/"
        with pytest.raises(KeyError):
            test_module.Config.parse(config_path)


def test_simulation_config():
    actual = test_module.Config.parse(
        str(TEST_DATA_DIR / 'simulation_config.json')
    )
    assert actual["target_simulator"] == "my_simulator"
    assert actual["network"] == str(Path(TEST_DATA_DIR / "circuit_config.json").resolve())
    assert actual["mechanisms_dir"] == str(Path(TEST_DATA_DIR / "../shared_components_mechanisms").resolve())
    assert actual["conditions"]["celsius"] == 34.0
    assert actual["conditions"]["v_init"] == -80


def test_dict_config():
    config = json.load(open(str(TEST_DATA_DIR / 'circuit_config.json'), "r"))
    # there are relative paths in the manifest you cannot resolve if you are using a dict
    with pytest.raises(BluepySnapError):
        test_module.Config(config)

    config["manifest"]["$NETWORK_DIR"] = str(TEST_DATA_DIR)
    config["manifest"]["$BASE_DIR"] = str(TEST_DATA_DIR)

    expected = test_module.Config.parse(str(TEST_DATA_DIR / 'circuit_config.json'))
    assert test_module.Config(config).resolve() == expected

    config = json.load(open(str(TEST_DATA_DIR / 'simulation_config.json'), "r"))
    # there are relative paths in the manifest you cannot resolve if you are using a dict and
    # the field "mechanisms_dir" is using a relative path
    with pytest.raises(BluepySnapError):
        test_module.Config(config)

    config["manifest"]["$OUTPUT_DIR"] = str(TEST_DATA_DIR / "reporting")
    config["manifest"]["$INPUT_DIR"] = str(TEST_DATA_DIR)

    # the field "mechanisms_dir" is still using a relative path
    with pytest.raises(BluepySnapError):
        test_module.Config(config).resolve()

    # does not allow Paths as values
    with pytest.raises(BluepySnapError):
        config = json.load(open(str(TEST_DATA_DIR / 'circuit_config.json'), "r"))
        config["manifest"]["$NETWORK_DIR"] = TEST_DATA_DIR
        config["manifest"]["$BASE_DIR"] = TEST_DATA_DIR
        test_module.Config.parse(config)
