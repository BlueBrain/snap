import json

import libsonata
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
from mock import Mock, PropertyMock, patch
from numpy import dtype
from pandas.api.types import is_categorical_dtype

import bluepysnap.nodes as test_module
from bluepysnap.bbp import Cell
from bluepysnap.circuit import Circuit
from bluepysnap.circuit_ids import CircuitNodeId, CircuitNodeIds
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.node_sets import NodeSets
from bluepysnap.sonata_constants import DEFAULT_NODE_TYPE, Node
from bluepysnap.utils import IDS_DTYPE

from utils import TEST_DATA_DIR, create_node_population


class TestNodes:
    def setup(self):
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
        }

    def test_property_value(self):
        assert self.test_obj.property_values("mtype") == {"L2_X", "L7_X", "L9_Z", "L8_Y", "L6_Y"}
        assert self.test_obj.property_values("other2") == {10, 11, 12, 13}

    def test_property_dtypes(self):
        expected = pd.Series(
            data=[
                dtype("int64"),
                dtype("O"),
                dtype("O"),
                dtype("O"),
                dtype("O"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("O"),
                dtype("int64"),
            ],
            index=[
                "layer",
                "model_template",
                "model_type",
                "morphology",
                "mtype",
                "rotation_angle_xaxis",
                "rotation_angle_yaxis",
                "rotation_angle_zaxis",
                "x",
                "y",
                "z",
                "@dynamics:holding_current",
                "other1",
                "other2",
            ],
        ).sort_index()
        pdt.assert_series_equal(self.test_obj.property_dtypes.sort_index(), expected)

    def test_property_dtypes_fail(self):
        a = pd.Series(
            data=[dtype("int64"), dtype("O")], index=["layer", "model_template"]
        ).sort_index()
        b = pd.Series(
            data=[dtype("int32"), dtype("O")], index=["layer", "model_template"]
        ).sort_index()

        with patch(
            "bluepysnap.nodes.NodePopulation.property_dtypes", new_callable=PropertyMock
        ) as mock:
            mock.side_effect = [a, b]
            circuit = Circuit(str(TEST_DATA_DIR / "circuit_config.json"))
            test_obj = test_module.Nodes(circuit)
            with pytest.raises(BluepySnapError):
                test_obj.property_dtypes.sort_index()

    def test_ids(self):
        np.random.seed(42)

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
        expected = CircuitNodeIds.from_arrays(["default2", "default2"], [3, 0], sort_index=False)
        assert tested == expected

        tested = self.test_obj.ids(limit=4)
        expected = CircuitNodeIds.from_dict({"default": [0, 1, 2], "default2": [0]})
        assert tested == expected

    def test_get(self):
        # return all properties for all the ids
        tested = self.test_obj.get()
        assert tested.shape == (self.test_obj.size, len(self.test_obj.property_names))

        # put NaN for the undefined values : only values for default2 in dropna
        assert len(tested.dropna()) == 4

        # the index of the dataframe is the index from all the NodeCircuitIds
        pdt.assert_index_equal(tested.index, self.test_obj.ids().index)

        # tested accessing data via circuit ids
        tested_ids = self.test_obj.ids({"population": "default"})
        assert tested.loc[tested_ids, "layer"].tolist() == [2, 6, 6]

        # tested columns
        tested = self.test_obj.get(properties=["other2", "other1", "layer"])
        assert tested.shape == (self.test_obj.size, 3)
        assert list(tested) == ["other2", "other1", "layer"]

        tested = self.test_obj.get(
            group={"population": "default2"}, properties=["other2", "other1", "layer"]
        )
        assert tested.shape == (4, 3)
        # correct ordering when setting the dataframe with the population dataframe
        assert tested.loc[("default2", 0)].tolist() == [10, "A", 7]
        with pytest.raises(KeyError):
            tested.loc[("default", 0)]

        tested = self.test_obj.get(
            group={"population": "default"}, properties=["other2", "other1", "layer"]
        )
        assert tested.shape == (3, 3)
        assert tested.loc[("default", 0)].tolist() == [np.NaN, np.NaN, 2]
        assert tested.loc[("default", 1)].tolist() == [np.NaN, np.NaN, 6]

        tested = self.test_obj.get(properties="layer")
        assert tested["layer"].tolist() == [2, 6, 6, 7, 8, 8, 2]

        tested = self.test_obj.get(properties="other2")
        assert tested["other2"].tolist() == [np.NaN, np.NaN, np.NaN, 10, 11, 12, 13]

        with pytest.raises(BluepySnapError):
            self.test_obj.get(properties=["other2", "unknown"])

        with pytest.raises(BluepySnapError):
            self.test_obj.get(properties="unknown")

    def test_h5_filepath(self):
        assert self.test_obj["default"].h5_filepath == str(TEST_DATA_DIR / "nodes.h5")

    def test_no_h5_filepath(self):
        test_obj = test_module.Nodes(Circuit(str(TEST_DATA_DIR / "circuit_config.json")))
        with patch("libsonata.NodeStorage.population_names") as patched:
            patched.return_value = []
            with pytest.raises(BluepySnapError, match="h5_filepath not found for population"):
                test_obj["default"].h5_filepath


class TestNodePopulation:
    def setup(self):
        self.test_obj = Circuit(str(TEST_DATA_DIR / "circuit_config.json")).nodes["default"]

    def test_basic(self):
        assert self.test_obj.name == "default"
        assert self.test_obj.size == 3
        assert self.test_obj.type == DEFAULT_NODE_TYPE
        assert sorted(self.test_obj.property_names) == [
            Cell.HOLDING_CURRENT,
            Cell.LAYER,
            Cell.MODEL_TEMPLATE,
            Cell.MODEL_TYPE,
            Cell.MORPHOLOGY,
            Cell.MTYPE,
            Cell.ROTATION_ANGLE_X,
            Cell.ROTATION_ANGLE_Y,
            Cell.ROTATION_ANGLE_Z,
            Cell.X,
            Cell.Y,
            Cell.Z,
        ]
        assert sorted(self.test_obj._node_sets) == sorted(
            json.load(open(str(TEST_DATA_DIR / "node_sets.json")))
        )

    def test_population_type(self):
        test_obj = create_node_population(
            str(TEST_DATA_DIR / "nodes.h5"),
            "default",
            node_sets=NodeSets(str(TEST_DATA_DIR / "node_sets.json")),
            pop_type="fake_type",
        )
        assert test_obj.type == "fake_type"

    def test_property_values(self):
        assert self.test_obj.property_values(Cell.LAYER) == {2, 6}
        assert self.test_obj.property_values(Cell.MORPHOLOGY) == {"morph-A", "morph-B", "morph-C"}
        test_obj_library = create_node_population(
            str(TEST_DATA_DIR / "nodes_with_library_small.h5"), "default"
        )
        assert test_obj_library.property_values("categorical") == {"A", "B", "C"}
        assert test_obj_library.property_values("categorical", is_present=True) == {"A", "B"}

    def test_property_dtypes(self):
        expected = pd.Series(
            data=[
                dtype("int64"),
                dtype("O"),
                dtype("O"),
                dtype("O"),
                dtype("O"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
            ],
            index=[
                "layer",
                "model_template",
                "model_type",
                "morphology",
                "mtype",
                "rotation_angle_xaxis",
                "rotation_angle_yaxis",
                "rotation_angle_zaxis",
                "x",
                "y",
                "z",
                "@dynamics:holding_current",
            ],
        ).sort_index()

        pdt.assert_series_equal(expected, self.test_obj.property_dtypes)

    def test_container_properties(self):
        expected = sorted(
            [
                "X",
                "Y",
                "Z",
                "MORPHOLOGY",
                "HOLDING_CURRENT",
                "ROTATION_ANGLE_X",
                "ROTATION_ANGLE_Y",
                "ROTATION_ANGLE_Z",
                "MTYPE",
                "LAYER",
                "MODEL_TEMPLATE",
                "MODEL_TYPE",
            ]
        )
        assert sorted(self.test_obj.container_property_names(Cell)) == expected
        expected = sorted(
            [
                "X",
                "Y",
                "Z",
                "MORPHOLOGY",
                "ROTATION_ANGLE_X",
                "ROTATION_ANGLE_Y",
                "ROTATION_ANGLE_Z",
                "MODEL_TEMPLATE",
                "MODEL_TYPE",
            ]
        )
        assert sorted(self.test_obj.container_property_names(Node)) == expected

        with pytest.raises(BluepySnapError):
            mapping = {"X": "x"}
            self.test_obj.container_property_names(mapping)

        with pytest.raises(BluepySnapError):
            self.test_obj.container_property_names(int)

    def test_as_edge_source_target(self):
        circuit = Circuit(str(TEST_DATA_DIR / "circuit_config.json"))
        assert circuit.nodes["default"].source_in_edges() == {"default", "default2"}
        assert circuit.nodes["default"].target_in_edges() == {"default", "default2"}

    def test_ids(self):
        _call = self.test_obj.ids
        npt.assert_equal(_call(), [0, 1, 2])
        npt.assert_equal(_call().dtype, IDS_DTYPE)
        npt.assert_equal(_call(group={}), [0, 1, 2])
        npt.assert_equal(_call(group=[]), [])
        npt.assert_equal(_call(limit=1), [0])
        # limit too big compared to the number of ids
        npt.assert_equal(_call(limit=15), [0, 1, 2])
        npt.assert_equal(len(_call(sample=2)), 2)
        # if sample > population.size --> sample = population.size
        npt.assert_equal(len(_call(sample=25)), 3)
        npt.assert_equal(_call(group=[], sample=2), [])
        npt.assert_equal(_call(group={Cell.MTYPE: "unknown"}, sample=2), [])
        npt.assert_equal(_call(0), [0])
        npt.assert_equal(_call(np.int64(0)), [0])
        npt.assert_equal(_call(np.uint64(0)), [0])
        npt.assert_equal(_call(np.int32(0)), [0])
        npt.assert_equal(_call(np.uint32(0)), [0])
        npt.assert_equal(_call([0, 1]), [0, 1])
        npt.assert_equal(_call([1, 0, 1]), [1, 0, 1])  # order and duplicates preserved
        npt.assert_equal(_call(np.array([1, 0, 1])), np.array([1, 0, 1]))

        # NodeCircuitId
        npt.assert_equal(_call(CircuitNodeId("default", 0)), [0])
        # List of NodeCircuitId
        npt.assert_equal(_call([CircuitNodeId("default", 0), CircuitNodeId("default", 1)]), [0, 1])
        # tuple of NodeCircuitId
        npt.assert_equal(_call((CircuitNodeId("default", 0), CircuitNodeId("default", 1))), [0, 1])
        # NodeCircuitId with wrong population
        npt.assert_equal(_call(CircuitNodeId("default2", 0)), [])
        npt.assert_equal(_call([CircuitNodeId("default2", 0), CircuitNodeId("default2", 1)]), [])
        # NodeCircuitId list with one wrong population and one ok
        npt.assert_equal(_call([CircuitNodeId("default2", 0), CircuitNodeId("default", 1)]), [1])

        # NodeCircuitIds
        ids = CircuitNodeIds.from_arrays(["default", "default"], [0, 1])
        npt.assert_equal(_call(ids), [0, 1])
        # returns only the ids for the default population
        ids = CircuitNodeIds.from_arrays(["default", "default", "default2"], [0, 1, 0])
        npt.assert_equal(_call(ids), [0, 1])
        # returns only the ids for the default population so should be []
        ids = CircuitNodeIds.from_arrays(["default2", "default2", "default2"], [0, 1, 2])
        npt.assert_equal(_call(ids), [])

        npt.assert_equal(_call({Cell.MTYPE: "L6_Y"}), [1, 2])
        npt.assert_equal(_call({Cell.X: (100, 203)}), [0, 1])
        npt.assert_equal(_call({Cell.MTYPE: "L6_Y", Cell.MORPHOLOGY: "morph-B"}), [1])

        npt.assert_equal(_call({"node_id": 1}), [1])
        npt.assert_equal(_call({"node_id": [1]}), [1])
        npt.assert_equal(_call({"node_id": [1, 2]}), [1, 2])
        npt.assert_equal(_call({"node_id": [1, 2, 42]}), [1, 2])
        npt.assert_equal(
            _call({"node_id": [1], "population": ["default"], Cell.MORPHOLOGY: "morph-B"}), [1]
        )

        # same query with a $and operator
        npt.assert_equal(_call({"$and": [{Cell.MTYPE: "L6_Y"}, {Cell.MORPHOLOGY: "morph-B"}]}), [1])
        npt.assert_equal(_call({Cell.MORPHOLOGY: ["morph-A", "morph-B"]}), [0, 1])
        npt.assert_equal(_call({"$and": [{}, {}]}), [0, 1, 2])
        npt.assert_equal(_call({"$and": [{}, {Cell.MORPHOLOGY: "morph-B"}]}), [1])
        # same query with a $or operator
        npt.assert_equal(
            _call({"$or": [{Cell.MORPHOLOGY: "morph-A"}, {Cell.MORPHOLOGY: "morph-B"}]}), [0, 1]
        )
        npt.assert_equal(
            _call({"$or": [{Cell.MTYPE: "L6_Y"}, {Cell.MORPHOLOGY: "morph-B"}]}), [1, 2]
        )
        npt.assert_equal(_call({"$or": [{}, {}]}), [0, 1, 2])
        npt.assert_equal(_call({"$or": [{}, {Cell.MORPHOLOGY: "morph-B"}]}), [0, 1, 2])
        # non destructive operation for queries
        query = {
            "$and": [
                {"$or": [{Cell.MTYPE: "L6_Y"}, {Cell.MORPHOLOGY: "morph-B"}]},
                {"node_id": [1]},
            ]
        }
        npt.assert_equal(_call(query), [1])
        npt.assert_equal(_call(query), [1])

        npt.assert_equal(_call("Layer2"), [0])
        npt.assert_equal(_call("Layer23"), [0])
        npt.assert_equal(_call("Empty_nodes"), [])
        npt.assert_equal(_call("Node2012"), [0, 1, 2])  # reordered + duplicates are removed
        npt.assert_equal(_call("Node12_L6_Y"), [1, 2])
        npt.assert_equal(_call("Node2_L6_Y"), [2])

        npt.assert_equal(_call("Node0_L6_Y"), [])  # return empty if disjoint samples
        npt.assert_equal(_call("Empty_L6_Y"), [])  # return empty if empty node_id = []
        npt.assert_equal(_call("Population_default"), [0, 1, 2])  # return all ids
        npt.assert_equal(_call("Population_default2"), [])  # return empty if diff population
        npt.assert_equal(_call("Population_default_L6_Y"), [1, 2])  # population + other query ok
        # population + other query + node_id ok
        npt.assert_equal(_call("Population_default_L6_Y_Node2"), [2])
        npt.assert_equal(_call("combined_Node0_L6_Y__Node12_L6_Y"), [1, 2])  # 'or' function
        npt.assert_equal(
            _call("combined_combined_Node0_L6_Y__Node12_L6_Y__"), [0, 1, 2]
        )  # imbricated '$or' functions

        npt.assert_equal(_call({"$node_set": "Node12_L6_Y", "node_id": 1}), [1])
        npt.assert_equal(_call({"$node_set": "Node12_L6_Y", "node_id": [1, 2, 3]}), [1, 2])
        npt.assert_equal(_call({"$node_set": "Node12_L6_Y", "population": "default"}), [1, 2])
        npt.assert_equal(
            _call({"$node_set": "Node12_L6_Y", "population": "default", "node_id": 1}), [1]
        )
        npt.assert_equal(_call({"$node_set": "Node12_L6_Y", Cell.MORPHOLOGY: "morph-B"}), [1])
        npt.assert_equal(
            _call(
                {
                    "$and": [
                        {"$node_set": "Node12_L6_Y", "population": "default"},
                        {Cell.MORPHOLOGY: "morph-B"},
                    ]
                }
            ),
            [1],
        )
        npt.assert_equal(
            _call(
                {
                    "$or": [
                        {"$node_set": "Node12_L6_Y", "population": "default"},
                        {Cell.MORPHOLOGY: "morph-B"},
                    ]
                }
            ),
            [1, 2],
        )

        with pytest.raises(BluepySnapError):
            _call("no-such-node-set")
        with pytest.raises(BluepySnapError):
            _call(-1)  # node ID out of range (lower boundary)
        with pytest.raises(BluepySnapError):
            _call([-1, 1])  # one of node IDs out of range (lower boundary)
        with pytest.raises(BluepySnapError):
            _call([1, -1])  # one of node IDs out of range, reversed order (lower boundary)
        with pytest.raises(BluepySnapError):
            _call(999)  # node ID out of range (upper boundary)
        with pytest.raises(BluepySnapError):
            _call([1, 999])  # one of node IDs out of range
        with pytest.raises(BluepySnapError):
            _call({"no-such-node-property": 42})
        with pytest.raises(BluepySnapError):
            _call({"$node_set": [1, 2]})
        with pytest.raises(BluepySnapError):
            _call({"$node_set": "no-such-node-set"})
        with pytest.raises(BluepySnapError):
            _call([CircuitNodeId("default", 1), CircuitNodeId("default2", 1), ("default2", 1)])

    def test_node_ids_by_filter_complex_query(self):
        test_obj = create_node_population(str(TEST_DATA_DIR / "nodes.h5"), "default")
        data = pd.DataFrame(
            {
                Cell.MTYPE: ["L23_MC", "L4_BP", "L6_BP", "L6_BPC"],
            }
        )
        # replace the data using the __dict__ directly
        test_obj.__dict__["_data"] = data

        # only full match is accepted
        npt.assert_equal(
            [1, 2],
            test_obj.ids(
                {
                    Cell.MTYPE: {"$regex": ".*BP"},
                }
            ),
        )
        # ...not 'startswith'
        npt.assert_equal([], test_obj.ids({Cell.MTYPE: {"$regex": "L6"}}))
        # ...or 'endswith'
        npt.assert_equal([], test_obj.ids({Cell.MTYPE: {"$regex": "BP"}}))
        # '$regex' is the only query modifier supported for the moment
        with pytest.raises(BluepySnapError) as e:
            test_obj.ids({Cell.MTYPE: {"err": ".*BP"}}, raise_missing_property=False)
        assert "Unknown query modifier" in e.value.args[0]
        with pytest.raises(BluepySnapError) as e:
            test_obj.ids({Cell.MTYPE: {"err": ".*BP"}})
        assert "Unknown node properties" in e.value.args[0]

    def test_get(self):
        _call = self.test_obj.get
        assert _call().shape == (3, 12)
        assert _call(0, Cell.MTYPE) == "L2_X"
        assert _call(CircuitNodeId("default", 0), Cell.MTYPE) == "L2_X"
        assert _call(np.int32(0), Cell.MTYPE) == "L2_X"
        pdt.assert_frame_equal(
            _call([1, 2], properties=[Cell.X, Cell.MTYPE, Cell.HOLDING_CURRENT]),
            pd.DataFrame(
                [
                    [201.0, "L6_Y", 0.2],
                    [301.0, "L6_Y", 0.3],
                ],
                columns=[Cell.X, Cell.MTYPE, Cell.HOLDING_CURRENT],
                index=[1, 2],
            ),
        )

        # NodeCircuitId same as [1, 2] for the default
        pdt.assert_frame_equal(
            _call(
                CircuitNodeIds.from_dict({"default": [1, 2]}),
                properties=[Cell.X, Cell.MTYPE, Cell.HOLDING_CURRENT],
            ),
            pd.DataFrame(
                [
                    [201.0, "L6_Y", 0.2],
                    [301.0, "L6_Y", 0.3],
                ],
                columns=[Cell.X, Cell.MTYPE, Cell.HOLDING_CURRENT],
                index=[1, 2],
            ),
        )

        # NodeCircuitId only consider the default population
        pdt.assert_frame_equal(
            _call(
                CircuitNodeIds.from_arrays(["default", "default", "default2"], [1, 2, 0]),
                properties=[Cell.X, Cell.MTYPE, Cell.HOLDING_CURRENT],
            ),
            pd.DataFrame(
                [
                    [201.0, "L6_Y", 0.2],
                    [301.0, "L6_Y", 0.3],
                ],
                columns=[Cell.X, Cell.MTYPE, Cell.HOLDING_CURRENT],
                index=[1, 2],
            ),
        )

        pdt.assert_frame_equal(
            _call("Node12_L6_Y", properties=[Cell.X, Cell.MTYPE, Cell.LAYER]),
            pd.DataFrame(
                [
                    [201.0, "L6_Y", 6],
                    [301.0, "L6_Y", 6],
                ],
                columns=[Cell.X, Cell.MTYPE, Cell.LAYER],
                index=[1, 2],
            ),
        )

        assert _call("Node0_L6_Y", properties=[Cell.X, Cell.MTYPE, Cell.LAYER]).empty
        assert _call(1, properties=[Cell.MTYPE]).tolist() == ["L6_Y"]
        assert _call([1], properties=Cell.MTYPE).tolist() == ["L6_Y"]
        assert _call([1, 2], properties=Cell.MTYPE).tolist() == ["L6_Y", "L6_Y"]
        with pytest.raises(BluepySnapError):
            _call(0, properties="no-such-property")
        with pytest.raises(BluepySnapError):
            _call(999)  # invalid node id
        with pytest.raises(BluepySnapError):
            _call([0, 999])  # one of node ids is invalid

    def test_get_with_library_small_number_of_values(self):
        test_obj = create_node_population(
            str(TEST_DATA_DIR / "nodes_with_library_small.h5"), "default"
        )
        assert test_obj.property_names == {"categorical", "string", "int", "float"}
        res = test_obj.get(properties=["categorical", "string", "int", "float"])
        assert is_categorical_dtype(res["categorical"])
        assert res["categorical"].tolist() == ["A", "A", "B", "A", "A", "A", "A"]
        assert res["categorical"].cat.categories.tolist() == ["A", "B", "C"]
        assert res["categorical"].cat.codes.tolist() == [0, 0, 1, 0, 0, 0, 0]
        assert res["string"].tolist() == ["AA", "BB", "CC", "DD", "EE", "FF", "GG"]
        assert res["int"].tolist() == [0, 0, 1, 0, 0, 0, 0]
        npt.assert_allclose(res["float"].tolist(), [0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 0.0])

    def test_get_with_library_large_number_of_values(self):
        test_obj = create_node_population(
            str(TEST_DATA_DIR / "nodes_with_library_large.h5"), "default"
        )
        assert test_obj.property_names == {"categorical", "string", "int", "float"}
        res = test_obj.get(properties=["categorical", "string", "int", "float"])
        assert not is_categorical_dtype(res["categorical"])
        assert res["categorical"].tolist() == ["A", "A", "B", "A"]
        assert res["string"].tolist() == ["AA", "BB", "CC", "DD"]
        assert res["int"].tolist() == [0, 0, 1, 0]
        npt.assert_allclose(res["float"].tolist(), [0.0, 0.0, 1.1, 0.0])

    def test_positions(self):
        _call = self.test_obj.positions
        expected = pd.Series([101.0, 102.0, 103.0], index=[Cell.X, Cell.Y, Cell.Z], name=0)
        pdt.assert_series_equal(_call(0), expected)
        pdt.assert_series_equal(_call(CircuitNodeId("default", 0)), expected)
        pdt.assert_frame_equal(
            _call([2, 0]),
            pd.DataFrame(
                [
                    [301.0, 302.0, 303.0],
                    [101.0, 102.0, 103.0],
                ],
                index=[2, 0],
                columns=[Cell.X, Cell.Y, Cell.Z],
            ),
        )

        # NodeCircuitIds
        pdt.assert_frame_equal(
            _call(CircuitNodeIds.from_arrays(["default", "default"], [2, 0], sort_index=False)),
            _call([2, 0]),
        )

    def test_orientations(self):
        _call = self.test_obj.orientations
        expected = [
            [0.738219, 0.0, 0.674560],
            [0.0, 1.0, 0.0],
            [-0.674560, 0.0, 0.738219],
        ]
        npt.assert_almost_equal(_call(0), expected, decimal=6)
        npt.assert_almost_equal(_call(CircuitNodeId("default", 0)), expected, decimal=6)
        pdt.assert_series_equal(
            _call([2, 0, 1]),
            pd.Series(
                [
                    np.array(
                        [
                            [0.462986, 0.0, 0.886365],
                            [0.0, 1.0, 0.0],
                            [-0.886365, 0.0, 0.462986],
                        ]
                    ),
                    np.array(
                        [
                            [0.738219, 0.0, 0.674560],
                            [0.0, 1.0, 0.0],
                            [-0.674560, 0.0, 0.738219],
                        ]
                    ),
                    np.array(
                        [
                            [-0.86768965, -0.44169042, 0.22808825],
                            [0.48942842, -0.8393853, 0.23641518],
                            [0.0870316, 0.31676788, 0.94450178],
                        ]
                    ),
                ],
                index=[2, 0, 1],
                name="orientation",
            ),
        )

        # NodeCircuitIds
        pdt.assert_series_equal(
            _call(
                CircuitNodeIds.from_arrays(
                    ["default", "default", "default"], [2, 0, 1], sort_index=False
                )
            ),
            _call([2, 0, 1]),
        )

        # NodePopulation without rotation_angle[x|z]
        _call_no_xz = create_node_population(
            str(TEST_DATA_DIR / "nodes_no_xz_rotation.h5"), "default"
        ).orientations
        # 0 and 2 node_ids have x|z rotation angles equal to zero
        npt.assert_almost_equal(_call_no_xz(0), _call(0))
        npt.assert_almost_equal(_call_no_xz(2), _call(2))
        npt.assert_almost_equal(
            _call_no_xz(1),
            [[0.97364046, -0.0, 0.22808825], [0.0, 1.0, -0.0], [-0.22808825, 0.0, 0.97364046]],
            decimal=6,
        )

        # NodePopulation without rotation_angle
        _call_no_rot = create_node_population(
            str(TEST_DATA_DIR / "nodes_no_rotation.h5"), "default"
        ).orientations

        pdt.assert_series_equal(
            _call_no_rot([2, 0, 1]),
            pd.Series([np.eye(3), np.eye(3), np.eye(3)], index=[2, 0, 1], name="orientation"),
        )

        # NodePopulation with quaternions
        _call_quat = create_node_population(
            str(TEST_DATA_DIR / "nodes_quaternions.h5"), "default"
        ).orientations

        npt.assert_almost_equal(
            _call_quat(0),
            [
                [1, 0.0, 0.0],
                [0.0, 0, -1.0],
                [0.0, 1.0, 0],
            ],
            decimal=6,
        )

        series = _call_quat([2, 0, 1])
        for i in range(len(series)):
            series.iloc[i] = np.around(series.iloc[i], decimals=1).astype(np.float64)

        pdt.assert_series_equal(
            series,
            pd.Series(
                [
                    np.array(
                        [
                            [0.0, -1.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ]
                    ),
                    np.array(
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 0.0, -1.0],
                            [0.0, 1.0, 0.0],
                        ]
                    ),
                    np.array(
                        [
                            [0.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0],
                            [-1.0, 0.0, 0.0],
                        ]
                    ),
                ],
                index=[2, 0, 1],
                name="orientation",
            ),
        )

        _call_missing_quat = create_node_population(
            str(TEST_DATA_DIR / "nodes_quaternions_w_missing.h5"), "default"
        ).orientations

        with pytest.raises(BluepySnapError):
            _call_missing_quat(0)

    def test_count(self):
        _call = self.test_obj.count
        assert _call(0) == 1
        assert _call([0, 1]) == 2
        assert _call(CircuitNodeIds.from_dict({"default": [0, 1]})) == 2
        assert _call({Cell.MTYPE: "L6_Y"}) == 2
        assert _call("Layer23") == 1

    def test_morph(self):
        from bluepysnap.morph import MorphHelper

        assert isinstance(self.test_obj.morph, MorphHelper)

    def test_models(self):
        from bluepysnap.neuron_models import NeuronModelsHelper

        assert isinstance(self.test_obj.models, NeuronModelsHelper)
