import json
from pathlib import Path

import pytest

import bluepysnap.config as test_module
from bluepysnap.exceptions import BluepySnapError

from utils import TEST_DATA_DIR, copy_config, edit_config


def parse(path):
    with open(path) as fd:
        config = json.load(fd)

    return test_module.Parser.parse(config, str(Path(path).parent))


def test_parse():
    actual = parse(str(TEST_DATA_DIR / "circuit_config.json"))

    # check double resolution and '.' works: $COMPONENT_DIR -> $BASE_DIR -> '.'
    assert actual["components"]["morphologies_dir"] == str(TEST_DATA_DIR / "morphologies")

    # check resolution and './' works: $NETWORK_DIR -> './'
    assert actual["networks"]["nodes"][0]["nodes_file"] == str(TEST_DATA_DIR / "nodes.h5")

    # check resolution of '../' works: $PARENT --> '../'
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["manifest"]["$PARENT"] = "../"
            config["components"]["other"] = "$PARENT/other"

        actual = parse(config_path)
        assert actual["components"]["other"] == str(Path(config_path.parent / "../other").resolve())

    # check resolution of '../' works in a path outside manifest
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["components"]["other"] = "../other"

        actual = parse(config_path)
        assert actual["components"]["other"] == str(Path(config_path.parent / "../other").resolve())

    # check resolution without manifest of '../' works in a path outside
    # i.e. : self.manifest contains the configdir even if manifest is not here
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            for k in list(config):
                config.pop(k)
            config["something"] = "../other"
        actual = parse(config_path)
        assert actual["something"] == str(Path(config_path.parent / "../other").resolve())

    # check resolution with multiple slashes
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["something"] = "$COMPONENT_DIR/something////else"
        actual = parse(config_path)
        assert actual["something"] == str(Path(config_path.parent) / "something" / "else")

    # check resolution with $ in a middle of the words
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["something"] = "$COMPONENT_DIR/somet$hing/else"
        actual = parse(config_path)
        assert actual["something"] == str(Path(config_path.parent) / "somet$hing" / "else")

    # check resolution with relative path without "." in the manifest
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["manifest"]["$NOPOINT"] = "nopoint"
            config["components"]["other"] = "$NOPOINT/other"
        actual = parse(config_path)
        assert actual["components"]["other"] == str(Path(config_path.parent) / "nopoint" / "other")

    # check resolution for non path objects
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            for k in list(config):
                config.pop(k)
            config["string"] = "string"
            config["int"] = 1
            config["double"] = 0.2
            # just to check because we use starting with '.' as a special case
            config["tricky_double"] = 0.2
            config["path"] = "./path"

        actual = parse(config_path)

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
            parse(config_path)

    # same but not in the manifest
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["components"]["other"] = "$COMPONENT_DIR/$BASE_DIR"
        with pytest.raises(BluepySnapError):
            parse(config_path)

    # manifest value not a string
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["manifest"]["$COMPONENT_DIR"] = 42
        with pytest.raises(BluepySnapError):
            parse(config_path)

    # relative path with an anchor in the middle is not allowed this breaks the purpose of the
    # anchors (they are not just generic placeholders)
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["components"]["other"] = "something/$COMPONENT_DIR/"
        with pytest.raises(BluepySnapError):
            parse(config_path)

    # abs path with an anchor in the middle is not allowed
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["components"]["other"] = "/something/$COMPONENT_DIR/"
        with pytest.raises(BluepySnapError):
            parse(config_path)

    # unknown anchor
    with copy_config() as config_path:
        with edit_config(config_path) as config:
            config["components"]["other"] = "$UNKNOWN/something/"
        with pytest.raises(KeyError):
            parse(config_path)


def test_simulation_config():
    actual = test_module.SimulationConfig.from_config(
        str(TEST_DATA_DIR / "simulation_config.json")
    ).to_dict()
    assert actual["target_simulator"] == "CORENEURON"
    assert actual["network"] == str(Path(TEST_DATA_DIR / "circuit_config.json").resolve())
    assert actual["mechanisms_dir"] == str(
        Path(TEST_DATA_DIR / "../shared_components_mechanisms").resolve()
    )
    assert actual["conditions"]["celsius"] == 34.0
    assert actual["conditions"]["v_init"] == -80


def test__resolve_population_configs():
    node = {
        "nodes_file": "nodes_path",
        "populations": {"node_pop": {"changed_node": "changed", "added_this": "node_test"}},
    }

    edge = {
        "edges_file": "edges_path",
        "populations": {"edge_pop": {"changed_edge": "changed", "added_this": "edge_test"}},
    }

    config = {
        "components": {
            "unchanged": True,
            "changed_node": "unchanged",
            "changed_edge": "unchanged",
        },
        "networks": {
            "nodes": [node],
            "edges": [edge],
        },
    }
    expected_node_pop = {
        "unchanged": True,
        "changed_node": "changed",
        "changed_edge": "unchanged",
        "added_this": "node_test",
        "nodes_file": "nodes_path",
    }
    expected_edge_pop = {
        "unchanged": True,
        "changed_node": "unchanged",
        "changed_edge": "changed",
        "added_this": "edge_test",
        "edges_file": "edges_path",
    }
    expected = {
        "nodes": {"node_pop": expected_node_pop},
        "edges": {"edge_pop": expected_edge_pop},
    }
    res = test_module.CircuitConfig._resolve_population_configs(config)

    assert res == expected
