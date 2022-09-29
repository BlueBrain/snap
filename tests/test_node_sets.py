import json
from unittest.mock import patch

import pytest

import bluepysnap.node_sets as test_module
from bluepysnap.exceptions import BluepySnapError

from utils import TEST_DATA_DIR


class TestNodeSets:
    def setup(self):
        self.test_obj = test_module.NodeSets(str(TEST_DATA_DIR / "node_sets_file.json"))

    def test_init(self):
        assert self.test_obj.content == {
            "double_combined": ["combined", "population_default_L6"],
            "Node2_L6_Y": {"mtype": ["L6_Y"], "node_id": [30, 20, 20]},
            "Layer23": {"layer": [3, 2, 2]},
            "population_default_L6": {"population": "default", "mtype": "L6_Y"},
            "combined": ["Node2_L6_Y", "Layer23"],
        }

        # node_id from Node2_L6_Y should be 2 and not [2], layer from Layer23 should be [2, 3]
        assert self.test_obj.resolved == {
            "Node2_L6_Y": {"mtype": "L6_Y", "node_id": [20, 30]},
            "Layer23": {"layer": [2, 3]},
            "combined": {"$or": [{"mtype": "L6_Y", "node_id": [20, 30]}, {"layer": [2, 3]}]},
            "population_default_L6": {"population": "default", "mtype": "L6_Y"},
            "double_combined": {
                "$or": [
                    {"$or": [{"mtype": "L6_Y", "node_id": [20, 30]}, {"layer": [2, 3]}]},
                    {"population": "default", "mtype": "L6_Y"},
                ]
            },
        }

    def test_get(self):
        assert self.test_obj["Node2_L6_Y"] == {"mtype": "L6_Y", "node_id": [20, 30]}
        assert self.test_obj["double_combined"] == {
            "$or": [
                {"$or": [{"mtype": "L6_Y", "node_id": [20, 30]}, {"layer": [2, 3]}]},
                {"population": "default", "mtype": "L6_Y"},
            ]
        }

    def test_iter(self):
        expected = set(json.loads((TEST_DATA_DIR / "node_sets_file.json").read_text()))
        assert set(self.test_obj) == expected


@patch("bluepysnap.utils.load_json")
def test_fail_resolve(mock_load):
    mock_load.return_value = {"empty_dict": {}}
    with pytest.raises(BluepySnapError):
        test_module.NodeSets(str(TEST_DATA_DIR / "node_sets_file.json"))

    mock_load.return_value = {"empty_list": []}
    with pytest.raises(BluepySnapError):
        test_module.NodeSets(str(TEST_DATA_DIR / "node_sets_file.json"))

    mock_load.return_value = {"int": 1}
    with pytest.raises(BluepySnapError):
        test_module.NodeSets(str(TEST_DATA_DIR / "node_sets_file.json"))

    mock_load.return_value = {"bool": True}
    with pytest.raises(BluepySnapError):
        test_module.NodeSets(str(TEST_DATA_DIR / "node_sets_file.json"))

    mock_load.return_value = {"combined": ["known", "unknown"], "known": {"v": 1}}
    with pytest.raises(BluepySnapError):
        test_module.NodeSets(str(TEST_DATA_DIR / "node_sets_file.json"))
