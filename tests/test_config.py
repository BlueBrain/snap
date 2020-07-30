import pytest
from pathlib2 import Path

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
    # not abs path in
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["manifest"]["$BASE_DIR"] = "not_absolute"
        with pytest.raises(BluepySnapError):
            test_module.Config.parse(config_path)

    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["manifest"]["$COMPONENT_DIR"] = "$BASE_DIR/$NETWORK_DIR"
        with pytest.raises(BluepySnapError):
            test_module.Config.parse(config_path)

    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["components"]["other"] = "$COMPONENT_DIR/$BASE_DIR"
        with pytest.raises(BluepySnapError):
            test_module.Config.parse(config_path)

    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["components"]["other"] = "something/$COMPONENT_DIR/"
        with pytest.raises(BluepySnapError):
            test_module.Config.parse(config_path)

    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["components"]["other"] = "/something/$COMPONENT_DIR/"
        with pytest.raises(BluepySnapError):
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
