import json
import re
from unittest.mock import patch

import libsonata
import pytest
from numpy.testing import assert_array_equal

import bluepysnap.node_sets as test_module
from bluepysnap.exceptions import BluepySnapError

from utils import TEST_DATA_DIR


class TestNodeSet:
    def setup_method(self):
        self.test_node_sets = test_module.NodeSets.from_file(
            str(TEST_DATA_DIR / "node_sets_file.json")
        )
        self.test_pop = libsonata.NodeStorage(str(TEST_DATA_DIR / "nodes.h5")).open_population(
            "default"
        )

    def test_get_ids(self):
        assert_array_equal(self.test_node_sets["Node2_L6_Y"].get_ids(self.test_pop), [])
        assert_array_equal(self.test_node_sets["double_combined"].get_ids(self.test_pop), [0, 1, 2])

        node_set = self.test_node_sets["failing"]

        with pytest.raises(BluepySnapError, match="No such attribute"):
            node_set.get_ids(self.test_pop)

        assert node_set.get_ids(self.test_pop, raise_missing_property=False) == []


class TestNodeSets:
    def setup_method(self):
        self.test_obj = test_module.NodeSets.from_file(str(TEST_DATA_DIR / "node_sets_file.json"))
        self.test_pop = libsonata.NodeStorage(str(TEST_DATA_DIR / "nodes.h5")).open_population(
            "default"
        )

    def test_init(self):
        assert self.test_obj.content == {
            "double_combined": ["combined", "population_default_L6"],
            "Node2_L6_Y": {"mtype": ["L6_Y"], "node_id": [30, 20, 20]},
            "Layer23": {"layer": [3, 2, 2]},
            "population_default_L6": {"population": "default", "mtype": "L6_Y"},
            "combined": ["Node2_L6_Y", "Layer23"],
            "failing": {"unknown_property": [0]},
        }

    def test_from_string(self):
        res = test_module.NodeSets.from_string(json.dumps(self.test_obj.content))
        assert res.content == self.test_obj.content

    def test_from_dict(self):
        res = test_module.NodeSets.from_dict(self.test_obj.content)
        assert res.content == self.test_obj.content

    def test_update(self):
        # update all keys
        tested = test_module.NodeSets.from_file(str(TEST_DATA_DIR / "node_sets_file.json"))
        res = tested.update(tested)

        # should return all keys as replaced
        assert res == {*self.test_obj}
        assert tested.content == self.test_obj.content

        # actually add a new node set
        additional = {"test": {"test_property": ["test_value"]}}
        res = tested.update(test_module.NodeSets.from_dict(additional))
        expected_content = {**self.test_obj.content, **additional}

        # None of the keys should be replaced
        assert res == set()
        assert tested.content == expected_content

        with pytest.raises(
            BluepySnapError, match=re.escape("Unexpected type: 'str' (expected: 'NodeSets')")
        ):
            tested.update("")

    def test_contains(self):
        assert "Layer23" in self.test_obj
        assert "find_me_you_will_not" not in self.test_obj
        with pytest.raises(BluepySnapError, match="Unexpected type"):
            42 in self.test_obj

    def test_getitem(self):
        assert isinstance(self.test_obj["Layer23"], test_module.NodeSet)

        with pytest.raises(BluepySnapError, match="Undefined node set:"):
            self.test_obj["no-such-node-set"]

    def test_iter(self):
        expected = set(json.loads((TEST_DATA_DIR / "node_sets_file.json").read_text()))
        assert set(self.test_obj) == expected
