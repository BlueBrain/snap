from unittest.mock import PropertyMock, patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest
from numpy import dtype

import bluepysnap.edges as test_module
from bluepysnap.bbp import Synapse
from bluepysnap.circuit import Circuit
from bluepysnap.circuit_ids import CircuitEdgeId, CircuitEdgeIds, CircuitNodeId, CircuitNodeIds
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.utils import IDS_DTYPE

from utils import TEST_DATA_DIR


class TestEdges:
    def setup_method(self):
        circuit = Circuit(str(TEST_DATA_DIR / "circuit_config.json"))
        self.test_obj = test_module.Edges(circuit)

    def test_get_population(self):
        assert isinstance(self.test_obj["default"], test_module.EdgePopulation)
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
        assert isinstance(values[0], test_module.EdgePopulation)
        assert values[0].name == "default"

        assert isinstance(values[1], test_module.EdgePopulation)
        assert values[1].name == "default2"

    def test_items(self):
        keys, values = zip(*self.test_obj.items())
        assert keys == ("default", "default2")
        assert isinstance(values[0], test_module.EdgePopulation)
        assert values[0].name == "default"

        assert isinstance(values[1], test_module.EdgePopulation)
        assert values[1].name == "default2"

    def test_size(self):
        assert self.test_obj.size == 8

    def test_property_names(self):
        assert self.test_obj.property_names == {
            "@dynamics:param1",
            "@source_node",
            "@target_node",
            "afferent_center_x",
            "afferent_center_y",
            "afferent_center_z",
            "afferent_section_id",
            "afferent_section_pos",
            "afferent_surface_x",
            "afferent_surface_y",
            "afferent_surface_z",
            "conductance",
            "delay",
            "efferent_center_x",
            "efferent_center_y",
            "efferent_center_z",
            "efferent_section_id",
            "efferent_section_pos",
            "efferent_surface_x",
            "efferent_surface_y",
            "efferent_surface_z",
            "other1",
            "other2",
            "syn_weight",
        }

    def test_property_dtypes(self):
        expected = pd.Series(
            data=[
                dtype("float32"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float32"),
                dtype("float64"),
                dtype("float32"),
                dtype("float64"),
                dtype("int64"),
                dtype("int64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float64"),
                dtype("float32"),
                dtype("float32"),
                dtype("float64"),
                dtype("float64"),
                IDS_DTYPE,
                IDS_DTYPE,
                dtype("O"),
                dtype("int32"),
            ],
            index=[
                "syn_weight",
                "@dynamics:param1",
                "afferent_surface_y",
                "afferent_surface_z",
                "conductance",
                "efferent_center_x",
                "delay",
                "afferent_center_z",
                "efferent_section_id",
                "afferent_section_id",
                "efferent_center_y",
                "afferent_center_x",
                "efferent_surface_z",
                "afferent_center_y",
                "afferent_surface_x",
                "efferent_surface_x",
                "afferent_section_pos",
                "efferent_section_pos",
                "efferent_surface_y",
                "efferent_center_z",
                "@source_node",
                "@target_node",
                "other1",
                "other2",
            ],
        ).sort_index()
        pdt.assert_series_equal(self.test_obj.property_dtypes.sort_index(), expected)

    def test_property_dtypes_fail(self):
        a = pd.Series(
            data=[dtype("int64"), dtype("float64")], index=["syn_weight", "efferent_surface_z"]
        ).sort_index()
        b = pd.Series(
            data=[dtype("int32"), dtype("float64")], index=["syn_weight", "efferent_surface_z"]
        ).sort_index()

        with patch(
            "bluepysnap.edges.EdgePopulation.property_dtypes", new_callable=PropertyMock
        ) as mock:
            mock.side_effect = [a, b]
            circuit = Circuit(str(TEST_DATA_DIR / "circuit_config.json"))
            test_obj = test_module.Edges(circuit)
            with pytest.raises(BluepySnapError):
                test_obj.property_dtypes.sort_index()

    def test_ids(self):
        np.random.seed(42)
        # single edge ID --> CircuitEdgeIds return populations with the 0 id
        expected = CircuitEdgeIds.from_tuples([("default", 0), ("default2", 0)])
        tested = self.test_obj.ids(0)
        assert tested == expected
        npt.assert_equal(tested.get_ids().dtype, IDS_DTYPE)
        tested = self.test_obj.ids(np.int64(0))
        assert tested == expected
        npt.assert_equal(tested.get_ids().dtype, IDS_DTYPE)
        tested = self.test_obj.ids(np.uint64(0))
        assert tested == expected
        npt.assert_equal(tested.get_ids().dtype, IDS_DTYPE)
        tested = self.test_obj.ids(np.int32(0))
        assert tested == expected
        npt.assert_equal(tested.get_ids().dtype, IDS_DTYPE)
        tested = self.test_obj.ids(np.uint32(0))
        assert tested == expected
        npt.assert_equal(tested.get_ids().dtype, IDS_DTYPE)

        # single edge ID list --> CircuitEdgeIds return populations with the 0 id
        expected = CircuitEdgeIds.from_tuples([("default", 0), ("default2", 0)])
        assert self.test_obj.ids([0]) == expected

        # default3 population does not exist and is asked explicitly
        with pytest.raises(BluepySnapError):
            ids = CircuitEdgeIds.from_arrays(["default", "default3"], [0, 3])
            self.test_obj.ids(ids)

        # seq of node ID --> CircuitEdgeIds return populations with the array of ids
        expected = CircuitEdgeIds.from_arrays(
            ["default", "default", "default2", "default2"], [0, 1, 0, 1]
        )
        tested = self.test_obj.ids([0, 1])
        assert tested == expected
        tested = self.test_obj.ids((0, 1))
        assert tested == expected
        tested = self.test_obj.ids(np.array([0, 1]))
        assert tested == expected

        ids = [CircuitEdgeId("default", 0), CircuitEdgeId("default", 1)]
        assert self.test_obj.ids(ids) == CircuitEdgeIds.from_dict({"default": [0, 1]})

        with pytest.raises(BluepySnapError):
            ids = [CircuitEdgeId("default", 0), ("default", 1)]
            self.test_obj.ids(ids)

        # Check operations on global ids
        ids = self.test_obj.ids([0, 1, 2, 3])
        assert ids.filter_population("default").append(ids.filter_population("default2")) == ids

        expected = CircuitEdgeIds.from_arrays(["default2", "default2"], [0, 1])
        assert ids.filter_population("default2").limit(2) == expected

        tested = self.test_obj.ids(sample=2)
        expected = CircuitEdgeIds.from_arrays(["default2", "default"], [2, 3], sort_index=False)
        assert tested == expected

        tested = self.test_obj.ids(limit=5)
        expected = CircuitEdgeIds.from_dict({"default": [0, 1, 2, 3], "default2": [0]})
        assert tested == expected

        with pytest.raises(BluepySnapError) as e:
            self.test_obj.ids({"afferent_center_i": (10, 11)})
        assert "Unknown edge properties: {'afferent_center_i'}" == e.value.args[0]

        tested = self.test_obj.ids({"afferent_center_x": (1110, 1110.5)})
        expected = CircuitEdgeIds.from_dict({"default": [0], "default2": [0]})
        assert tested == expected

        tested = self.test_obj.ids(
            {"afferent_center_x": (1111, 1112), "efferent_center_z": (2132, 2134)}
        )
        expected = CircuitEdgeIds.from_dict({"default": [2], "default2": [2]})
        assert tested == expected

        tested = self.test_obj.ids(
            {"$and": [{"@dynamics:param1": (0, 2)}, {"afferent_surface_x": (1211, 1211)}]}
        )
        expected = CircuitEdgeIds.from_dict({"default": [1], "default2": [1]})
        assert tested == expected

        tested = self.test_obj.ids({"$or": [{"@dynamics:param1": (0, 2)}, {"@source_node": [0]}]})
        expected = CircuitEdgeIds.from_dict({"default": [0, 1, 2], "default2": [0, 1, 2]})
        assert tested == expected

        tested = self.test_obj.ids({"population": ["default2"], "afferent_center_x": (1113, 1114)})
        expected = CircuitEdgeIds.from_dict({"default2": [3]})
        assert tested == expected

        tested = self.test_obj.ids({"population": ["default3"], "afferent_center_x": (1113, 1114)})
        expected = CircuitEdgeIds.from_arrays([], [])
        assert tested == expected

        tested = self.test_obj.ids({"population": ["default", "default2"], "@target_node": [1]})
        expected = CircuitEdgeIds.from_dict({"default": [1, 2, 3], "default2": [1, 2, 3]})
        assert tested == expected

    def test_get(self):
        with pytest.raises(BluepySnapError, match="You need to set edge_ids in get."):
            self.test_obj.get(properties=["other2"])

        ids = CircuitEdgeIds.from_dict({"default": [0, 1, 2, 3], "default2": [0, 1, 2, 3]})
        with pytest.deprecated_call(
            match="Returning ids with get/properties is deprecated and will be removed in 1.0.0"
        ):
            tested = self.test_obj.get(ids, None)
        assert tested == ids

        tested = self.test_obj.get(ids, properties=self.test_obj.property_names)
        assert len(tested) == 8
        assert len(list(tested)) == 24

        # put NaN for the undefined values : only values for default2 in dropna
        assert len(tested.dropna()) == 4

        # the index of the dataframe is indentical to the CircuitEdgeIds index
        pdt.assert_index_equal(tested.index, ids.index)
        pdt.assert_frame_equal(
            self.test_obj.get([0, 1, 2, 3], properties=self.test_obj.property_names), tested
        )

        # tested columns
        tested = self.test_obj.get(ids, properties=["other2", "other1", "@source_node"])
        expected = pd.DataFrame(
            {
                "other2": np.array([np.NaN, np.NaN, np.NaN, np.NaN, 10, 11, 12, 13], dtype=float),
                "other1": np.array(
                    [np.NaN, np.NaN, np.NaN, np.NaN, "A", "B", "C", "D"], dtype=object
                ),
                "@source_node": np.array([2, 0, 0, 2, 2, 0, 0, 2], dtype=int),
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("default", 0),
                    ("default", 1),
                    ("default", 2),
                    ("default", 3),
                    ("default2", 0),
                    ("default2", 1),
                    ("default2", 2),
                    ("default2", 3),
                ],
                names=["population", "edge_ids"],
            ),
        )
        pdt.assert_frame_equal(tested, expected)

        tested = self.test_obj.get(
            CircuitEdgeIds.from_dict({"default2": [0, 1, 2, 3]}),
            properties=["other2", "other1", "@source_node"],
        )
        # correct ordering when setting the dataframe with the population dataframe
        expected = pd.DataFrame(
            {
                "other2": np.array([10, 11, 12, 13], dtype=np.int32),
                "other1": np.array(["A", "B", "C", "D"], dtype=object),
                "@source_node": np.array([2, 0, 0, 2], dtype=int),
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("default2", 0),
                    ("default2", 1),
                    ("default2", 2),
                    ("default2", 3),
                ],
                names=["population", "edge_ids"],
            ),
        )
        pdt.assert_frame_equal(tested, expected)

        with pytest.raises(KeyError, match="'default'"):
            tested.loc[("default", 0)]

        tested = self.test_obj.get(
            CircuitEdgeIds.from_dict({"default": [0, 1, 2, 3]}),
            properties=["other2", "other1", "@source_node"],
        )
        expected = pd.DataFrame(
            {
                "other2": np.array([np.NaN, np.NaN, np.NaN, np.NaN], dtype=float),
                "other1": np.array([np.NaN, np.NaN, np.NaN, np.NaN], dtype=object),
                "@source_node": np.array([2, 0, 0, 2], dtype=int),
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("default", 0),
                    ("default", 1),
                    ("default", 2),
                    ("default", 3),
                ],
                names=["population", "edge_ids"],
            ),
        )
        pdt.assert_frame_equal(tested, expected)

        tested = self.test_obj.get(ids, properties="@source_node")
        expected = pd.DataFrame(
            {
                "@source_node": np.array([2, 0, 0, 2, 2, 0, 0, 2], dtype=int),
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("default", 0),
                    ("default", 1),
                    ("default", 2),
                    ("default", 3),
                    ("default2", 0),
                    ("default2", 1),
                    ("default2", 2),
                    ("default2", 3),
                ],
                names=["population", "edge_ids"],
            ),
        )
        pdt.assert_frame_equal(tested, expected)

        tested = self.test_obj.get(ids, properties="other2")
        expected = pd.DataFrame(
            {
                "other2": np.array([np.NaN, np.NaN, np.NaN, np.NaN, 10, 11, 12, 13], dtype=float),
            },
            index=pd.MultiIndex.from_tuples(
                [
                    ("default", 0),
                    ("default", 1),
                    ("default", 2),
                    ("default", 3),
                    ("default2", 0),
                    ("default2", 1),
                    ("default2", 2),
                    ("default2", 3),
                ],
                names=["population", "edge_ids"],
            ),
        )
        pdt.assert_frame_equal(tested, expected)

        with pytest.raises(BluepySnapError, match="Unknown properties required: {'unknown'}"):
            self.test_obj.get(ids, properties=["other2", "unknown"])

        with pytest.raises(BluepySnapError, match="Unknown properties required: {'unknown'}"):
            self.test_obj.get(ids, properties="unknown")

        with pytest.deprecated_call(
            match=(
                "Returning ids with get/properties is deprecated and will be removed in 1.0.0. "
                "Please use Edges.ids instead."
            )
        ):
            self.test_obj.get(ids)

    def test_properties_deprecated(self):
        ids = CircuitEdgeIds.from_dict({"default": [0, 1, 2, 3], "default2": [0, 1, 2, 3]})
        with pytest.deprecated_call(
            match="Edges.properties function is deprecated and will be removed in 1.0.0"
        ):
            tested = self.test_obj.properties(ids, properties=["other2", "@source_node"])
        expected = self.test_obj.get(ids, properties=["other2", "@source_node"])
        pdt.assert_frame_equal(tested, expected, check_exact=False)

    def test_afferent_nodes(self):
        assert self.test_obj.afferent_nodes(0) == CircuitNodeIds.from_arrays(["default"], [2])
        assert self.test_obj.afferent_nodes(np.int64(0)) == CircuitNodeIds.from_arrays(
            ["default"], [2]
        )
        assert self.test_obj.afferent_nodes(np.uint64(0)) == CircuitNodeIds.from_arrays(
            ["default"], [2]
        )
        assert self.test_obj.afferent_nodes(np.int32(0)) == CircuitNodeIds.from_arrays(
            ["default"], [2]
        )
        assert self.test_obj.afferent_nodes(np.int32(0)) == CircuitNodeIds.from_arrays(
            ["default"], [2]
        )
        assert self.test_obj.afferent_nodes(
            CircuitNodeId("default", 0)
        ) == CircuitNodeIds.from_arrays(["default"], [2])
        assert self.test_obj.afferent_nodes([0, 1]) == CircuitNodeIds.from_dict({"default": [2, 0]})
        ids = CircuitNodeIds.from_dict({"default": [0, 1], "default2": [0, 1]})
        assert self.test_obj.afferent_nodes(ids) == CircuitNodeIds.from_dict({"default": [2, 0]})
        assert self.test_obj.afferent_nodes(0, unique=False) == CircuitNodeIds.from_arrays(
            ["default", "default"], [2, 2]
        )

        # use global mapping for nodes
        assert self.test_obj.afferent_nodes({"other1": "A"}) == CircuitNodeIds.from_arrays([], [])
        assert self.test_obj.afferent_nodes({"mtype": "L6_Y"}) == CircuitNodeIds.from_dict(
            {"default": [0, 2]}
        )

    def test_efferent_nodes(self):
        assert self.test_obj.efferent_nodes(0) == CircuitNodeIds.from_arrays(["default"], [1])
        assert self.test_obj.efferent_nodes(np.int64(0)) == CircuitNodeIds.from_arrays(
            ["default"], [1]
        )
        assert self.test_obj.efferent_nodes(np.uint64(0)) == CircuitNodeIds.from_arrays(
            ["default"], [1]
        )
        assert self.test_obj.efferent_nodes(np.int32(0)) == CircuitNodeIds.from_arrays(
            ["default"], [1]
        )
        assert self.test_obj.efferent_nodes(np.uint32(0)) == CircuitNodeIds.from_arrays(
            ["default"], [1]
        )
        assert self.test_obj.efferent_nodes(
            CircuitNodeId("default", 0)
        ) == CircuitNodeIds.from_arrays(["default"], [1])
        assert self.test_obj.efferent_nodes([0, 2]) == CircuitNodeIds.from_dict({"default": [0, 1]})
        ids = CircuitNodeIds.from_dict({"default": [0, 2]})
        assert self.test_obj.efferent_nodes(ids) == CircuitNodeIds.from_dict({"default": [1, 0]})
        assert self.test_obj.efferent_nodes(0, unique=False) == CircuitNodeIds.from_arrays(
            ["default", "default"], [1, 1]
        )

        # use global mapping for nodes
        assert self.test_obj.efferent_nodes({"other1": "A"}) == CircuitNodeIds.from_arrays([], [])
        assert self.test_obj.efferent_nodes({"mtype": "L6_Y"}) == CircuitNodeIds.from_dict(
            {"default": [0, 1]}
        )

    def test_pathway_edges(self):
        properties = [Synapse.AXONAL_DELAY]
        source = CircuitNodeIds.from_dict({"default": [0, 1]})
        target = CircuitNodeIds.from_dict({"default": [1, 2]})

        expected_index = CircuitEdgeIds.from_dict({"default": [1, 2], "default2": [1, 2]})
        pdt.assert_frame_equal(
            self.test_obj.pathway_edges(source=source, target=target, properties=properties),
            pd.DataFrame(
                [
                    [88.1862],
                    [52.1881],
                    [88.1862],
                    [52.1881],
                ],
                columns=properties,
                index=expected_index.index,
            ),
            check_dtype=False,
        )

        properties = [Synapse.SOURCE_NODE_ID, "other1"]
        expected_index = CircuitEdgeIds.from_dict({"default": [1, 2], "default2": [1, 2]})
        pdt.assert_frame_equal(
            self.test_obj.pathway_edges(source=source, target=target, properties=properties),
            pd.DataFrame(
                [
                    [0, np.nan],
                    [0, np.nan],
                    [0, "B"],
                    [0, "C"],
                ],
                columns=properties,
                index=expected_index.index,
            ),
            check_dtype=False,
        )

        # without the properties should return the CircuitEdgeIds
        assert self.test_obj.pathway_edges(source, target) == expected_index
        assert self.test_obj.pathway_edges(source, target, None) == expected_index

        # without the properties and the target
        assert self.test_obj.pathway_edges(source, None) == CircuitEdgeIds.from_dict(
            {"default": [1, 2], "default2": [1, 2]}
        )
        # without the properties and the source
        assert self.test_obj.pathway_edges(None, source) == CircuitEdgeIds.from_dict(
            {"default": [0, 1, 2, 3], "default2": [0, 1, 2, 3]}
        )

        # raise if both source and target are not set
        with pytest.raises(BluepySnapError):
            self.test_obj.pathway_edges(None, None, None)

        # test with simple CircuitNodeId
        properties = [Synapse.SOURCE_NODE_ID, Synapse.TARGET_NODE_ID]
        source = CircuitNodeId("default", 0)
        target = CircuitNodeId("default", 1)
        expected_index = CircuitEdgeIds.from_dict({"default": [1, 2], "default2": [1, 2]})
        pdt.assert_frame_equal(
            self.test_obj.pathway_edges(source=source, target=target, properties=properties),
            pd.DataFrame(
                [
                    [0, 1],
                    [0, 1],
                    [0, 1],
                    [0, 1],
                ],
                columns=properties,
                index=expected_index.index,
            ),
            check_dtype=False,
        )

        # use global mapping for nodes
        assert self.test_obj.pathway_edges(
            source={"mtype": "L6_Y"}, target={"mtype": "L2_X"}
        ) == CircuitEdgeIds.from_tuples([("default", 0), ("default2", 0)])

    def test_afferent_edges(self):
        # without the properties
        target = CircuitNodeIds.from_dict({"default": [0, 1]})
        assert self.test_obj.afferent_edges(target, None) == CircuitEdgeIds.from_dict(
            {"default": [0, 1, 2, 3], "default2": [0, 1, 2, 3]}
        )

        # with a single int
        expected = CircuitEdgeIds.from_dict({"default": [1, 2, 3], "default2": [1, 2, 3]})
        assert self.test_obj.afferent_edges(1, None) == expected

        # with a CircuitNodeId
        assert self.test_obj.afferent_edges(CircuitNodeId("default", 1), None) == expected

        properties = [Synapse.AXONAL_DELAY]
        pdt.assert_frame_equal(
            self.test_obj.afferent_edges(1, properties),
            pd.DataFrame(
                [
                    [88.1862],
                    [52.1881],
                    [11.1058],
                    [88.1862],
                    [52.1881],
                    [11.1058],
                ],
                columns=properties,
                index=expected.index,
            ),
            check_dtype=False,
        )

        # with an undefined other1 field for the population default
        properties = [Synapse.SOURCE_NODE_ID, "other1"]
        expected_index = CircuitEdgeIds.from_dict(
            {"default": [0, 1, 2, 3], "default2": [0, 1, 2, 3]}
        )
        pdt.assert_frame_equal(
            self.test_obj.afferent_edges(
                CircuitNodeIds.from_dict({"default": [0, 1]}), properties=properties
            ),
            pd.DataFrame(
                [
                    [2, np.nan],
                    [0, np.nan],
                    [0, np.nan],
                    [2, np.nan],
                    [2, "A"],
                    [0, "B"],
                    [0, "C"],
                    [2, "D"],
                ],
                columns=properties,
                index=expected_index.index,
            ),
            check_dtype=False,
        )

    def test_efferent_edges(self):
        target = CircuitNodeIds.from_dict({"default": [2]})
        expected = CircuitEdgeIds.from_dict({"default": [0, 3], "default2": [0, 3]})
        assert self.test_obj.efferent_edges(target, None) == expected
        assert self.test_obj.efferent_edges(2, None) == expected
        assert self.test_obj.efferent_edges(CircuitNodeId("default", 2), None) == expected

        properties = [Synapse.AXONAL_DELAY]
        pdt.assert_frame_equal(
            self.test_obj.efferent_edges(2, properties),
            pd.DataFrame(
                [
                    [99.8945],
                    [11.1058],
                    [99.8945],
                    [11.1058],
                ],
                columns=properties,
                index=expected.index,
            ),
            check_dtype=False,
        )

        # with an undefined other1 field for the population default
        properties = [Synapse.TARGET_NODE_ID, "other1"]
        expected_index = CircuitEdgeIds.from_dict({"default": [0, 3], "default2": [0, 3]})

        pdt.assert_frame_equal(
            self.test_obj.efferent_edges(2, properties),
            pd.DataFrame(
                [
                    [0, np.nan],
                    [1, np.nan],
                    [0, "A"],
                    [1, "D"],
                ],
                columns=properties,
                index=expected_index.index,
            ),
            check_dtype=False,
        )

    def test_pair_edges(self):
        # no connection between 0 and 2
        assert self.test_obj.pair_edges(0, 2, None) == CircuitEdgeIds.from_arrays([], [])
        actual = self.test_obj.pair_edges(0, 2, [Synapse.AXONAL_DELAY])
        assert actual.empty

        assert self.test_obj.pair_edges(2, 0, None) == CircuitEdgeIds.from_tuples(
            [("default", 0), ("default2", 0)]
        )

        properties = [Synapse.AXONAL_DELAY]
        pdt.assert_frame_equal(
            self.test_obj.pair_edges(2, 0, properties),
            pd.DataFrame(
                [
                    [99.8945],
                    [99.8945],
                ],
                columns=properties,
                index=CircuitEdgeIds.from_tuples([("default", 0), ("default2", 0)]).index,
            ),
            check_dtype=False,
        )

    def test_iter_connections(self):
        ids = CircuitNodeIds.from_dict({"default": [0, 1, 2], "default2": [0, 1, 2]})
        # ordered by target
        expected = [
            (CircuitNodeId("default", 2), CircuitNodeId("default", 0)),
            (CircuitNodeId("default", 0), CircuitNodeId("default", 1)),
            (CircuitNodeId("default", 2), CircuitNodeId("default", 1)),
            (CircuitNodeId("default", 2), CircuitNodeId("default", 0)),
            (CircuitNodeId("default", 0), CircuitNodeId("default", 1)),
            (CircuitNodeId("default", 2), CircuitNodeId("default", 1)),
        ]
        for i, tested in enumerate(self.test_obj.iter_connections(source=ids, target=ids)):
            assert tested == expected[i]

        for i, tested in enumerate(self.test_obj.iter_connections(source=None, target=ids)):
            assert tested == expected[i]

        # same but ordered by source
        expected = [
            (CircuitNodeId("default", 0), CircuitNodeId("default", 1)),
            (CircuitNodeId("default", 2), CircuitNodeId("default", 0)),
            (CircuitNodeId("default", 2), CircuitNodeId("default", 1)),
            (CircuitNodeId("default", 0), CircuitNodeId("default", 1)),
            (CircuitNodeId("default", 2), CircuitNodeId("default", 0)),
            (CircuitNodeId("default", 2), CircuitNodeId("default", 1)),
        ]
        for i, tested in enumerate(self.test_obj.iter_connections(source=ids, target=None)):
            assert tested == expected[i]

        expected = [
            (
                CircuitNodeId("default", 2),
                CircuitNodeId("default", 0),
                CircuitEdgeIds.from_dict({"default": [0]}),
            ),
            (
                CircuitNodeId("default", 0),
                CircuitNodeId("default", 1),
                CircuitEdgeIds.from_dict({"default": [1, 2]}),
            ),
            (
                CircuitNodeId("default", 2),
                CircuitNodeId("default", 1),
                CircuitEdgeIds.from_dict({"default": [3]}),
            ),
            (
                CircuitNodeId("default", 2),
                CircuitNodeId("default", 0),
                CircuitEdgeIds.from_dict({"default2": [0]}),
            ),
            (
                CircuitNodeId("default", 0),
                CircuitNodeId("default", 1),
                CircuitEdgeIds.from_dict({"default2": [1, 2]}),
            ),
            (
                CircuitNodeId("default", 2),
                CircuitNodeId("default", 1),
                CircuitEdgeIds.from_dict({"default2": [3]}),
            ),
        ]
        for i, tested in enumerate(
            self.test_obj.iter_connections(source=ids, target=ids, return_edge_ids=True)
        ):
            assert tested == expected[i]

        expected = [
            (CircuitNodeId("default", 2), CircuitNodeId("default", 0), 1),
            (CircuitNodeId("default", 0), CircuitNodeId("default", 1), 2),
            (CircuitNodeId("default", 2), CircuitNodeId("default", 1), 1),
            (CircuitNodeId("default", 2), CircuitNodeId("default", 0), 1),
            (CircuitNodeId("default", 0), CircuitNodeId("default", 1), 2),
            (CircuitNodeId("default", 2), CircuitNodeId("default", 1), 1),
        ]
        for i, tested in enumerate(
            self.test_obj.iter_connections(source=ids, target=ids, return_edge_count=True)
        ):
            assert tested == expected[i]

        with pytest.raises(BluepySnapError):
            next(
                self.test_obj.iter_connections(
                    ids, ids, return_edge_ids=True, return_edge_count=True
                )
            )
