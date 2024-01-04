import pickle
from unittest.mock import PropertyMock, patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
from numpy import dtype

import bluepysnap.nodes as test_module
from bluepysnap.circuit import Circuit
from bluepysnap.circuit_ids import CircuitNodeIds
from bluepysnap.circuit_ids_types import IDS_DTYPE, CircuitNodeId
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.node_sets import NodeSets

from utils import PICKLED_SIZE_ADJUSTMENT, TEST_DATA_DIR


class TestNodes:
    def setup_method(self):
        circuit = Circuit(str(TEST_DATA_DIR / "circuit_config.json"))
        self.test_obj = test_module.Nodes(circuit)

    def test_get_population(self):
        assert isinstance(self.test_obj["default"], test_module.NodePopulation)
        with pytest.raises(BluepySnapError):
            self.test_obj["unknown"]

    def test_iter(self):
        assert sorted(self.test_obj) == ["default", "default2"]

    def test_population_names(self):
        assert self.test_obj.population_names == ["default", "default2"]

    def test_keys_names(self):
        assert list(self.test_obj.keys()) == ["default", "default2"]

    def test_values_population(self):
        values = list(self.test_obj.values())
        assert isinstance(values[0], test_module.NodePopulation)
        assert values[0].name == "default"

        assert isinstance(values[1], test_module.NodePopulation)
        assert values[1].name == "default2"

    def test_items(self):
        keys, values = zip(*self.test_obj.items())
        assert keys == ("default", "default2")
        assert isinstance(values[0], test_module.NodePopulation)
        assert values[0].name == "default"

        assert isinstance(values[1], test_module.NodePopulation)
        assert values[1].name == "default2"

    def test_size(self):
        assert self.test_obj.size == 7

    def test_property_names(self):
        assert self.test_obj.property_names == {
            "rotation_angle_zaxis",
            "y",
            "layer",
            "mtype",
            "model_type",
            "z",
            "x",
            "rotation_angle_yaxis",
            "morphology",
            "rotation_angle_xaxis",
            "model_template",
            "other1",
            "other2",
            "@dynamics:holding_current",
            "@dynamics:input_resistance",
        }

    def test_property_value(self):
        assert self.test_obj.property_values("mtype") == {"L2_X", "L7_X", "L9_Z", "L8_Y", "L6_Y"}
        assert self.test_obj.property_values("other2") == {10, 11, 12, 13}

    def test_ids(self):
        np.random.seed(0)

        # None --> CircuitNodeIds with all ids
        tested = self.test_obj.ids()
        expected = CircuitNodeIds.from_dict({"default": [0, 1, 2], "default2": [0, 1, 2, 3]})
        assert tested == expected
        npt.assert_equal(tested.get_ids().dtype, IDS_DTYPE)

        # CircuitNodeIds --> CircuitNodeIds and check if the population and node ids exist
        ids = CircuitNodeIds.from_arrays(["default", "default2"], [0, 3])
        tested = self.test_obj.ids(ids)
        assert tested == ids

        # default3 population does not exist and is asked explicitly
        with pytest.raises(BluepySnapError):
            ids = CircuitNodeIds.from_arrays(["default", "default3"], [0, 3])
            self.test_obj.ids(ids)

        # (default2, 5) does not exist and is asked explicitly
        with pytest.raises(BluepySnapError):
            ids = CircuitNodeIds.from_arrays(["default", "default2"], [0, 5])
            self.test_obj.ids(ids)

        # single node ID --> CircuitNodeIds return populations with the 0 id
        expected = CircuitNodeIds.from_arrays(["default", "default2"], [0, 0])
        tested = self.test_obj.ids(0)
        assert tested == expected

        # single node ID --> CircuitNodeIds raise if the ID is not in all population
        with pytest.raises(BluepySnapError):
            self.test_obj.ids(3)

        # seq of node ID --> CircuitNodeIds return populations with the array of ids
        expected = CircuitNodeIds.from_arrays(
            ["default", "default", "default2", "default2"], [0, 1, 0, 1]
        )
        tested = self.test_obj.ids([0, 1])
        assert tested == expected
        tested = self.test_obj.ids((0, 1))
        assert tested == expected
        tested = self.test_obj.ids(np.array([0, 1]))
        assert tested == expected

        ids = [CircuitNodeId("default", 0), CircuitNodeId("default", 1)]
        assert self.test_obj.ids(ids) == CircuitNodeIds.from_dict({"default": [0, 1]})

        with pytest.raises(BluepySnapError):
            ids = [CircuitNodeId("default", 0), ("default", 1)]
            self.test_obj.ids(ids)

        # seq node ID --> CircuitNodeIds raise if on ID is not in all populations
        with pytest.raises(BluepySnapError):
            self.test_obj.ids([0, 1, 2, 3])

        # node sets
        assert self.test_obj.ids("Layer2") == CircuitNodeIds.from_arrays(
            ["default", "default2"], [0, 3]
        )
        assert self.test_obj.ids("Layer23") == CircuitNodeIds.from_arrays(
            ["default", "default2"], [0, 3]
        )
        assert self.test_obj.ids("Population_default_L6_Y_Node2") == CircuitNodeIds.from_arrays(
            ["default"], [2]
        )
        assert self.test_obj.ids(
            "combined_combined_Node0_L6_Y__Node12_L6_Y__"
        ) == CircuitNodeIds.from_arrays(["default", "default", "default", "default2"], [0, 1, 2, 3])

        # Mapping --> CircuitNodeIds query on the populations empty dict return all
        assert self.test_obj.ids({}) == self.test_obj.ids()

        assert self.test_obj.ids({"$and": []}) == self.test_obj.ids()
        assert self.test_obj.ids({"$or": []}) == CircuitNodeIds.from_dict({})

        # Mapping --> CircuitNodeIds query on the populations
        tested = self.test_obj.ids({"layer": 2})
        expected = CircuitNodeIds.from_arrays(["default", "default2"], [0, 3])
        assert tested == expected

        # Mapping --> CircuitNodeIds query on the populations no raise if not in one of the pop
        tested = self.test_obj.ids({"other1": ["A", "D"]})
        expected = CircuitNodeIds.from_arrays(["default2", "default2"], [0, 3])
        assert tested == expected

        # Mapping --> CircuitNodeIds query on the populations no raise if not in one of the pop
        tested = self.test_obj.ids({"other1": ["A", "D"], "layer": 2})
        expected = CircuitNodeIds.from_arrays(["default2"], [3])
        assert tested == expected

        # Mapping --> CircuitNodeIds query on the populations no raise if not in one of the pop
        tested = self.test_obj.ids({"$or": [{"other1": ["A", "D"]}, {"layer": 2}]})
        expected = CircuitNodeIds.from_arrays(["default", "default2", "default2"], [0, 0, 3])
        assert tested == expected

        # Mapping --> CircuitNodeIds query on the populations no raise if not in one of the pop
        tested = self.test_obj.ids({"$and": [{"other1": ["A", "D"]}, {"layer": 2}]})
        expected = CircuitNodeIds.from_arrays(["default2"], [3])
        assert tested == expected

        # Mapping --> CircuitNodeIds query on the population node ids with mapping.
        # single pop
        tested = self.test_obj.ids({"population": "default"})
        expected = self.test_obj.ids().filter_population("default")
        assert tested == expected
        # multiple pop
        tested = self.test_obj.ids({"population": ["default", "default2"]})
        expected = self.test_obj.ids()
        assert tested == expected
        # not existing pop (should not raise)
        tested = self.test_obj.ids({"population": "default4"})
        expected = CircuitNodeIds.from_arrays([], [])
        assert tested == expected

        # single pop and node ids
        tested = self.test_obj.ids({"population": ["default"], "node_id": [1, 2]})
        expected = CircuitNodeIds.from_arrays(["default", "default"], [1, 2])
        assert tested == expected
        # single pop and node ids with not present node id (should not raise)
        tested = self.test_obj.ids({"population": ["default"], "node_id": [1, 5]})
        expected = CircuitNodeIds.from_arrays(["default"], [1])
        assert tested == expected
        # not existing node ids (should not raise)
        tested = self.test_obj.ids({"population": ["default"], "node_id": [5, 6, 7]})
        expected = CircuitNodeIds.from_arrays([], [])
        assert tested == expected

        # multiple pop and node ids
        tested = self.test_obj.ids({"population": ["default", "default2"], "node_id": [1, 0]})
        expected = CircuitNodeIds.from_arrays(
            ["default", "default", "default2", "default2"], [1, 0, 1, 0]
        )
        assert tested == expected
        # multiple pop and node ids with not present node id (should not raise)
        tested = self.test_obj.ids({"population": ["default", "default2"], "node_id": [1, 0, 3]})
        expected = CircuitNodeIds.from_arrays(
            ["default", "default", "default2", "default2", "default2"], [1, 0, 1, 0, 3]
        )
        assert tested == expected

        # Check operations on global ids
        ids = self.test_obj.ids()
        assert ids.filter_population("default").append(ids.filter_population("default2")) == ids

        expected = CircuitNodeIds.from_arrays(["default2", "default2"], [0, 1])
        assert ids.filter_population("default2").limit(2) == expected

        tested = self.test_obj.ids(sample=2)
        expected = CircuitNodeIds.from_arrays(["default2", "default"], [3, 2], sort_index=False)
        assert tested == expected

        tested = self.test_obj.ids(limit=4)
        expected = CircuitNodeIds.from_dict({"default": [0, 1, 2], "default2": [0]})
        assert tested == expected

    def test_get(self):
        # return all properties for all the ids
        tested = self.test_obj.get()
        tested = pd.concat(df for _, df in tested)
        assert tested.shape == (self.test_obj.size, len(self.test_obj.property_names))

        # put NaN for the undefined values :
        # empty: "other1", "other2" missing in default and "input_resistance" in "default2'
        assert len(tested.dropna()) == 0

        cols = tested.columns.difference(["@dynamics:input_resistance"])
        assert len(tested[cols].dropna()) == 4

        cols = tested.columns.difference(["other1", "other2"])
        assert len(tested[cols].dropna()) == 3

        # the index of the dataframe is the index from all the NodeCircuitIds
        pdt.assert_index_equal(tested.index, self.test_obj.ids().index)

        # tested accessing data via circuit ids
        tested_ids = self.test_obj.ids({"population": "default"})
        assert tested.loc[tested_ids, "layer"].tolist() == [2, 6, 6]

        # tested columns
        tested = self.test_obj.get(properties=["other2", "other1", "layer"])
        tested = pd.concat(df for _, df in tested)
        expected = pd.DataFrame(
            {
                "other2": np.array([np.NaN, np.NaN, np.NaN, 10, 11, 12, 13], dtype=float),
                "other1": np.array([np.NaN, np.NaN, np.NaN, "A", "B", "C", "D"], dtype=object),
                "layer": np.array([2, 6, 6, 7, 8, 8, 2], dtype=int),
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("default", 0),
                    ("default", 1),
                    ("default", 2),
                    ("default2", 0),
                    ("default2", 1),
                    ("default2", 2),
                    ("default2", 3),
                ],
                names=["population", "node_ids"],
            ),
        )
        pdt.assert_frame_equal(tested[expected.columns], expected)

        tested = self.test_obj.get(
            group={"population": "default2"}, properties=["other2", "other1", "layer"]
        )
        expected = pd.DataFrame(
            {
                "other2": np.array([10, 11, 12, 13], dtype=int),
                "other1": np.array(["A", "B", "C", "D"], dtype=object),
                "layer": np.array([7, 8, 8, 2], dtype=int),
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("default2", 0),
                    ("default2", 1),
                    ("default2", 2),
                    ("default2", 3),
                ],
                names=["population", "node_ids"],
            ),
        )
        tested = pd.concat(df for _, df in tested)
        pdt.assert_frame_equal(tested, expected)

        with pytest.raises(KeyError, match="'default'"):
            tested.loc[("default", 0)]

        tested = self.test_obj.get(
            group={"population": "default"}, properties=["other2", "other1", "layer"]
        )
        expected = pd.DataFrame(
            {
                "layer": np.array([2, 6, 6], dtype=int),
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("default", 0),
                    ("default", 1),
                    ("default", 2),
                ],
                names=["population", "node_ids"],
            ),
        )
        tested = pd.concat(df for _, df in tested)
        pdt.assert_frame_equal(tested, expected)

        tested = self.test_obj.get(properties="layer")
        expected = pd.DataFrame(
            {
                "layer": np.array([2, 6, 6, 7, 8, 8, 2], dtype=int),
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("default", 0),
                    ("default", 1),
                    ("default", 2),
                    ("default2", 0),
                    ("default2", 1),
                    ("default2", 2),
                    ("default2", 3),
                ],
                names=["population", "node_ids"],
            ),
        )
        tested = pd.concat(df for _, df in tested)
        pdt.assert_frame_equal(tested, expected)

        tested = self.test_obj.get(properties="other2")
        expected = pd.DataFrame(
            {
                "other2": np.array([np.NaN, np.NaN, np.NaN, 10, 11, 12, 13], dtype=float),
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("default", 0),
                    ("default", 1),
                    ("default", 2),
                    ("default2", 0),
                    ("default2", 1),
                    ("default2", 2),
                    ("default2", 3),
                ],
                names=["population", "node_ids"],
            ),
        )
        tested = pd.concat(df for _, df in tested)
        pdt.assert_frame_equal(tested, expected)

        with pytest.raises(BluepySnapError, match="Unknown properties required: {'unknown'}"):
            next(self.test_obj.get(properties=["other2", "unknown"]))

        with pytest.raises(BluepySnapError, match="Unknown properties required: {'unknown'}"):
            next(self.test_obj.get(properties="unknown"))

    def test_functionality_with_separate_node_set(self):
        with pytest.raises(BluepySnapError, match="Undefined node set"):
            self.test_obj.ids("ExtraLayer2")

        node_sets = NodeSets.from_file(str(TEST_DATA_DIR / "node_sets_extra.json"))

        assert self.test_obj.ids(node_sets["ExtraLayer2"]) == CircuitNodeIds.from_arrays(
            ["default", "default2"], [0, 3]
        )

        with pytest.raises(BluepySnapError, match="Undefined node set"):
            next(self.test_obj.get("ExtraLayer2"))

        tested = pd.concat(df for _, df in self.test_obj.get(node_sets["ExtraLayer2"]))
        expected = pd.concat(df for _, df in self.test_obj.get("Layer2"))
        pdt.assert_frame_equal(tested, expected)

    def test_pickle(self, tmp_path):
        pickle_path = tmp_path / "pickle.pkl"

        # trigger some cached properties, to makes sure they aren't being pickeld
        self.test_obj.size
        self.test_obj.property_names

        with open(pickle_path, "wb") as fd:
            pickle.dump(self.test_obj, fd)

        with open(pickle_path, "rb") as fd:
            test_obj = pickle.load(fd)

        assert pickle_path.stat().st_size < 100 + PICKLED_SIZE_ADJUSTMENT
        assert test_obj.size == 7
