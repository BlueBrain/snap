import json
from unittest.mock import patch

import libsonata
import pytest
from numpy.testing import assert_array_equal

import bluepysnap.node_sets as test_module
from bluepysnap.exceptions import BluepySnapError

from utils import TEST_DATA_DIR


class TestNodeSets:
    def setup_method(self):
        self.test_obj = test_module.NodeSets(str(TEST_DATA_DIR / "node_sets_file.json"))
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

    def test_get_ids(self):
        assert_array_equal(self.test_obj.get_ids("Node2_L6_Y", self.test_pop), [])
        assert_array_equal(
            self.test_obj.get_ids("double_combined", self.test_pop),
            [0, 1, 2],
        )
        with pytest.raises(BluepySnapError, match="No such attribute"):
            self.test_obj.get_ids("failing", self.test_pop)

        assert self.test_obj.get_ids("failing", self.test_pop, raise_missing_property=False) == []

    def test_iter(self):
        expected = set(json.loads((TEST_DATA_DIR / "node_sets_file.json").read_text()))
        assert set(self.test_obj) == expected
