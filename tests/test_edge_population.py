import pickle
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

import bluepysnap.edges.edge_population as test_module
from bluepysnap.bbp import Synapse
from bluepysnap.circuit import Circuit
from bluepysnap.circuit_ids import CircuitEdgeIds, CircuitNodeIds
from bluepysnap.circuit_ids_types import IDS_DTYPE, CircuitEdgeId, CircuitNodeId
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.sonata_constants import DEFAULT_EDGE_TYPE, Edge

from utils import PICKLED_SIZE_ADJUSTMENT, TEST_DATA_DIR, copy_test_data, edit_config


def index_as_ids_dtypes(values):
    """have pandas index types match"""
    return np.array(values, dtype=IDS_DTYPE)


class TestEdgePopulation:
    @staticmethod
    def get_edge_population(circuit_path, pop_name):
        circuit = Circuit(circuit_path)
        return test_module.EdgePopulation(circuit, pop_name)

    def setup_method(self):
        self.test_obj = TestEdgePopulation.get_edge_population(
            TEST_DATA_DIR / "circuit_config.json", "default"
        )

    def test_basic(self):
        assert self.test_obj.name == "default"
        assert self.test_obj.source.name == "default"
        assert self.test_obj.target.name == "default"
        assert self.test_obj.size, 4
        assert sorted(self.test_obj.property_names) == sorted(
            [
                Synapse.SOURCE_NODE_ID,
                Synapse.TARGET_NODE_ID,
                Synapse.AXONAL_DELAY,
                Synapse.G_SYNX,
                Synapse.POST_X_CENTER,
                Synapse.POST_Y_CENTER,
                Synapse.POST_Z_CENTER,
                Synapse.POST_X_SURFACE,
                Synapse.POST_Y_SURFACE,
                Synapse.POST_Z_SURFACE,
                Synapse.PRE_X_CENTER,
                Synapse.PRE_Y_CENTER,
                Synapse.PRE_Z_CENTER,
                Synapse.PRE_X_SURFACE,
                Synapse.PRE_Y_SURFACE,
                Synapse.PRE_Z_SURFACE,
                Synapse.POST_SECTION_ID,
                Synapse.POST_SECTION_POS,
                Synapse.PRE_SECTION_ID,
                Synapse.PRE_SECTION_POS,
                Synapse.SYN_WEIGHT,
                test_module.DYNAMICS_PREFIX + "param1",
            ]
        )
        assert self.test_obj.type == DEFAULT_EDGE_TYPE

    def test_population_type(self):
        with copy_test_data() as (config_dir, config_path):
            with edit_config(config_path) as config:
                config["networks"]["edges"] = [
                    {
                        "edge_types_file": None,
                        "edges_file": str(Path(config_dir) / "edges_complete_graph.h5"),
                        "populations": {"default": {"type": "fake_type"}},
                    }
                ]

            test_obj = TestEdgePopulation.get_edge_population(config_path, "default")

            assert test_obj.type == "fake_type"

    def test_container_properties(self):
        expected = sorted(
            [
                "PRE_Y_SURFACE",
                "PRE_Z_SURFACE",
                "PRE_X_CENTER",
                "POST_Y_CENTER",
                "AXONAL_DELAY",
                "POST_X_CENTER",
                "POST_Y_SURFACE",
                "POST_Z_SURFACE",
                "PRE_Y_CENTER",
                "POST_Z_CENTER",
                "PRE_Z_CENTER",
                "PRE_X_SURFACE",
                "POST_X_SURFACE",
                "POST_SECTION_ID",
                "PRE_SECTION_ID",
                "POST_SECTION_POS",
                "PRE_SECTION_POS",
                "SYN_WEIGHT",
                "SOURCE_NODE_ID",
                "TARGET_NODE_ID",
            ]
        )
        assert sorted(self.test_obj.container_property_names(Edge)) == expected
        with pytest.raises(BluepySnapError):
            mapping = {"X": "x"}
            self.test_obj.container_property_names(mapping)

        with pytest.raises(BluepySnapError):
            self.test_obj.container_property_names(int)

    def test_nodes(self):
        assert self.test_obj._nodes("default").name == "default"
        with pytest.raises(BluepySnapError):
            self.test_obj._nodes("no-such-population")

    def test_property_dtypes(self):
        from numpy import dtype

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
                dtype("int64"),
                dtype("int64"),
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
            ],
        ).sort_index()

        pdt.assert_series_equal(expected, self.test_obj.property_dtypes)

    def test_ids(self):
        tested = self.test_obj.ids()
        npt.assert_equal(tested, np.array([0, 1, 2, 3]))
        npt.assert_equal(tested.dtype, IDS_DTYPE)

        assert self.test_obj.ids(0) == [0]
        assert self.test_obj.ids(np.int64(0)) == [0]
        assert self.test_obj.ids(np.uint64(0)) == [0]
        assert self.test_obj.ids(np.int32(0)) == [0]
        assert self.test_obj.ids(np.uint32(0)) == [0]
        npt.assert_equal(self.test_obj.ids([0, 1]), np.array([0, 1]))
        npt.assert_equal(self.test_obj.ids(np.array([0, 1])), np.array([0, 1]))
        npt.assert_equal(self.test_obj.ids(CircuitEdgeId("default", 0)), [0])
        npt.assert_equal(self.test_obj.ids(CircuitEdgeId("default2", 0)), [])
        ids = CircuitEdgeIds.from_tuples([("default", 0), ("default", 1)])
        npt.assert_equal(self.test_obj.ids(ids), np.array([0, 1]))
        ids = CircuitEdgeIds.from_tuples([("default", 0), ("default2", 1)])
        npt.assert_equal(self.test_obj.ids(ids), np.array([0]))
        ids = CircuitEdgeIds.from_tuples([("default2", 0), ("default2", 1)])
        npt.assert_equal(self.test_obj.ids(ids), [])
        npt.assert_equal(self.test_obj.ids(), np.array([0, 1, 2, 3]))

        # limit too big compared to the number of ids
        npt.assert_equal(self.test_obj.ids(limit=15), [0, 1, 2, 3])
        npt.assert_equal(len(self.test_obj.ids(sample=2)), 2)
        # if sample > population.size --> sample = population.size
        npt.assert_equal(len(self.test_obj.ids(sample=25)), 4)

        # check iterables in queries
        npt.assert_equal(self.test_obj.ids({"@target_node": [0, 1]}), [0, 1, 2, 3])
        npt.assert_equal(self.test_obj.ids({"@target_node": (0, 1)}), [0, 1, 2, 3])
        npt.assert_equal(self.test_obj.ids({"@target_node": map(int, [0, 1])}), [0, 1, 2, 3])

    def test_get_1(self):
        properties = [
            Synapse.PRE_GID,
            Synapse.POST_GID,
            Synapse.AXONAL_DELAY,
            Synapse.POST_X_CENTER,
            test_module.DYNAMICS_PREFIX + "param1",
        ]
        edge_ids = [0, 1]
        actual = self.test_obj.get(edge_ids, properties)
        expected = pd.DataFrame(
            [
                (2, 0, 99.8945, 1110.0, 0.0),
                (0, 1, 88.1862, 1111.0, 1.0),
            ],
            columns=properties,
            index=index_as_ids_dtypes(edge_ids),
        )
        pdt.assert_frame_equal(actual, expected, check_dtype=False)

    def test_get_2(self):
        prop = Synapse.AXONAL_DELAY
        edge_ids = [1, 0]
        actual = self.test_obj.get(edge_ids, prop)
        expected = pd.Series([88.1862, 99.8945], index=index_as_ids_dtypes(edge_ids), name=prop)
        pdt.assert_series_equal(actual, expected, check_dtype=False)

    def test_get_3(self):
        properties = [Synapse.PRE_GID, Synapse.AXONAL_DELAY]
        pdt.assert_series_equal(
            self.test_obj.get([], properties[0]), pd.Series(name=properties[0], dtype=np.float64)
        )
        pdt.assert_frame_equal(self.test_obj.get([], properties), pd.DataFrame(columns=properties))

    def test_get_4(self):
        with pytest.raises(BluepySnapError):
            self.test_obj.get([0], "no-such-property")

    def test_get_without_properties(self):
        edge_ids = [0, 1]
        actual = self.test_obj.get(edge_ids, None)
        expected = np.asarray(edge_ids, dtype=np.int64)
        npt.assert_equal(actual, expected)

    def test_get_all_edge_ids_types(self):
        assert self.test_obj.get(0, Synapse.PRE_GID).tolist() == [2]
        assert self.test_obj.get(np.int64(0), Synapse.PRE_GID).tolist() == [2]
        assert self.test_obj.get(np.uint64(0), Synapse.PRE_GID).tolist() == [2]
        assert self.test_obj.get(np.int32(0), Synapse.PRE_GID).tolist() == [2]
        assert self.test_obj.get(np.uint32(0), Synapse.PRE_GID).tolist() == [2]

        assert self.test_obj.get([0], Synapse.PRE_GID).tolist() == [2]
        assert self.test_obj.get([0, 1], Synapse.PRE_GID).tolist() == [2, 0]
        assert self.test_obj.get(CircuitEdgeId("default", 0), Synapse.PRE_GID).tolist() == [2]
        assert self.test_obj.get(
            CircuitEdgeIds.from_tuples([("default", 0)]), Synapse.PRE_GID
        ).tolist() == [2]
        assert self.test_obj.get(CircuitEdgeId("default2", 0), Synapse.PRE_GID).tolist() == []
        assert self.test_obj.get(
            CircuitEdgeIds.from_tuples([("default", 0), ("default", 1)]), Synapse.PRE_GID
        ).tolist() == [2, 0]
        assert self.test_obj.get(
            CircuitEdgeIds.from_tuples([("default", 0), ("default2", 1)]), Synapse.PRE_GID
        ).tolist() == [2]
        assert (
            self.test_obj.get(
                CircuitEdgeIds.from_tuples([("default2", 0), ("default2", 1)]), Synapse.PRE_GID
            ).tolist()
            == []
        )

    def test_positions_1(self):
        actual = self.test_obj.positions([0], "afferent", "center")
        expected = pd.DataFrame(
            [[1110.0, 1120.0, 1130.0]], index=index_as_ids_dtypes([0]), columns=["x", "y", "z"]
        )
        pdt.assert_frame_equal(actual, expected)
        pdt.assert_frame_equal(self.test_obj.positions(0, "afferent", "center"), actual)
        pdt.assert_frame_equal(self.test_obj.positions(np.int64(0), "afferent", "center"), actual)
        pdt.assert_frame_equal(self.test_obj.positions(np.uint64(0), "afferent", "center"), actual)
        pdt.assert_frame_equal(self.test_obj.positions(np.int32(0), "afferent", "center"), actual)
        pdt.assert_frame_equal(self.test_obj.positions(np.uint32(0), "afferent", "center"), actual)

    def test_positions_2(self):
        actual = self.test_obj.positions([1], "afferent", "surface")
        expected = pd.DataFrame(
            [[1211.0, 1221.0, 1231.0]], index=index_as_ids_dtypes([1]), columns=["x", "y", "z"]
        )
        pdt.assert_frame_equal(actual, expected)

    def test_positions_3(self):
        actual = self.test_obj.positions([2], "efferent", "center")
        expected = pd.DataFrame(
            [[2112.0, 2122.0, 2132.0]], index=index_as_ids_dtypes([2]), columns=["x", "y", "z"]
        )
        pdt.assert_frame_equal(actual, expected)

    def test_positions_4(self):
        actual = self.test_obj.positions([3], "efferent", "surface")
        expected = pd.DataFrame(
            [[2213.0, 2223.0, 2233.0]], index=index_as_ids_dtypes([3]), columns=["x", "y", "z"]
        )
        pdt.assert_frame_equal(actual, expected)

    def test_positions_5(self):
        with pytest.raises(AssertionError):
            self.test_obj.positions([2], "err", "center")

    def test_positions_6(self):
        with pytest.raises(AssertionError):
            self.test_obj.positions([2], "afferent", "err")

    def test_afferent_nodes(self):
        tested = self.test_obj.afferent_nodes(0)
        npt.assert_equal(tested, [2])
        npt.assert_equal(tested.dtype, IDS_DTYPE)

        tested = self.test_obj.afferent_nodes(np.int64(0))
        npt.assert_equal(tested, [2])
        npt.assert_equal(tested.dtype, IDS_DTYPE)

        tested = self.test_obj.afferent_nodes(np.uint64(0))
        npt.assert_equal(tested, [2])
        npt.assert_equal(tested.dtype, IDS_DTYPE)

        tested = self.test_obj.afferent_nodes(np.int32(0))
        npt.assert_equal(tested, [2])
        npt.assert_equal(tested.dtype, IDS_DTYPE)

        tested = self.test_obj.afferent_nodes(np.uint32(0))
        npt.assert_equal(tested, [2])
        npt.assert_equal(tested.dtype, IDS_DTYPE)

        npt.assert_equal(self.test_obj.afferent_nodes(0, unique=False), [2])

        npt.assert_equal(self.test_obj.afferent_nodes(1), [0, 2])
        npt.assert_equal(self.test_obj.afferent_nodes(1, unique=False), [0, 0, 2])

        npt.assert_equal(self.test_obj.afferent_nodes(2), [])

        npt.assert_equal(self.test_obj.afferent_nodes([0, 1]), [0, 2])
        npt.assert_equal(self.test_obj.afferent_nodes([0, 1], unique=False), [2, 0, 0, 2])

        npt.assert_equal(self.test_obj.afferent_nodes({}), [0, 2])
        npt.assert_equal(
            self.test_obj.afferent_nodes({"mtype": "L2_X"}), [2]
        )  # eq node id 0 as target
        npt.assert_equal(
            self.test_obj.afferent_nodes({"mtype": "L2_X"}), [2]
        )  # eq node id 0 as target

        npt.assert_equal(self.test_obj.afferent_nodes(None), [0, 2])
        npt.assert_equal(self.test_obj.afferent_nodes(None, unique=False), [2, 0, 0, 2])

    def test_efferent_nodes(self):
        tested = self.test_obj.efferent_nodes(0)
        npt.assert_equal(tested, [1])
        npt.assert_equal(tested.dtype, IDS_DTYPE)

        tested = self.test_obj.efferent_nodes(np.int64(0))
        npt.assert_equal(tested, [1])
        npt.assert_equal(tested.dtype, IDS_DTYPE)

        tested = self.test_obj.efferent_nodes(np.uint64(0))
        npt.assert_equal(tested, [1])
        npt.assert_equal(tested.dtype, IDS_DTYPE)

        tested = self.test_obj.efferent_nodes(np.int32(0))
        npt.assert_equal(tested, [1])
        npt.assert_equal(tested.dtype, IDS_DTYPE)

        tested = self.test_obj.efferent_nodes(np.uint32(0))
        npt.assert_equal(tested, [1])
        npt.assert_equal(tested.dtype, IDS_DTYPE)

        npt.assert_equal(self.test_obj.efferent_nodes(0, unique=False), [1, 1])

        npt.assert_equal(self.test_obj.efferent_nodes(1), [])
        npt.assert_equal(self.test_obj.efferent_nodes(1, unique=False), [])

        npt.assert_equal(self.test_obj.efferent_nodes(2), [0, 1])
        npt.assert_equal(self.test_obj.efferent_nodes(2, unique=False), [0, 1])

        npt.assert_equal(self.test_obj.efferent_nodes([0, 1]), [1])
        npt.assert_equal(self.test_obj.efferent_nodes([0, 1], unique=False), [1, 1])

        npt.assert_equal(self.test_obj.efferent_nodes({}), [0, 1])
        npt.assert_equal(
            self.test_obj.efferent_nodes({"mtype": "L2_X"}), [1]
        )  # eq node id 0 as source
        npt.assert_equal(
            self.test_obj.efferent_nodes({"mtype": "L2_X"}), [1]
        )  # eq node id 0 as source

        npt.assert_equal(self.test_obj.efferent_nodes(None), [0, 1])
        npt.assert_equal(self.test_obj.efferent_nodes(None, unique=False), [0, 1, 1, 1])

    def test_afferent_edges(self):
        tested = self.test_obj.afferent_edges([0, 1], None)
        npt.assert_equal(tested, [0, 1, 2, 3])
        npt.assert_equal(tested.dtype, IDS_DTYPE)

    def test_afferent_edges_1(self):
        npt.assert_equal(self.test_obj.afferent_edges(1, None), [1, 2, 3])

        npt.assert_equal(self.test_obj.afferent_edges(np.int64(1), None), [1, 2, 3])

        npt.assert_equal(self.test_obj.afferent_edges(np.uint64(1), None), [1, 2, 3])

        npt.assert_equal(self.test_obj.afferent_edges(np.int32(1), None), [1, 2, 3])

        npt.assert_equal(self.test_obj.afferent_edges(np.uint32(1), None), [1, 2, 3])

    def test_afferent_edges_2(self):
        properties = [Synapse.AXONAL_DELAY]
        pdt.assert_frame_equal(
            self.test_obj.afferent_edges(1, properties),
            pd.DataFrame(
                [
                    [88.1862],
                    [52.1881],
                    [11.1058],
                ],
                columns=properties,
                index=index_as_ids_dtypes([1, 2, 3]),
            ),
            check_dtype=False,
        )

    def test_efferent_edges_1(self):
        tested = self.test_obj.efferent_edges(2, None)
        npt.assert_equal(tested, [0, 3])
        npt.assert_equal(tested.dtype, IDS_DTYPE)

    def test_efferent_edges_2(self):
        properties = [Synapse.AXONAL_DELAY]
        pdt.assert_frame_equal(
            self.test_obj.efferent_edges(2, properties),
            pd.DataFrame(
                [
                    [99.8945],
                    [11.1058],
                ],
                columns=properties,
                index=index_as_ids_dtypes([0, 3]),
            ),
            check_dtype=False,
        )

    def test_pair_edges_1(self):
        tested = self.test_obj.pair_edges(0, 2, None)
        npt.assert_equal(tested, [])
        npt.assert_equal(tested.dtype, IDS_DTYPE)

    def test_pair_edges_2(self):
        actual = self.test_obj.pair_edges(0, 2, [Synapse.AXONAL_DELAY])
        assert actual.empty

    def test_pair_edges_3(self):
        tested = self.test_obj.pair_edges(2, 0, None)
        npt.assert_equal(tested, [0])
        npt.assert_equal(tested.dtype, IDS_DTYPE)

    def test_pair_edges_4(self):
        properties = [Synapse.AXONAL_DELAY]
        pdt.assert_frame_equal(
            self.test_obj.pair_edges(2, 0, properties),
            pd.DataFrame(
                [
                    [99.8945],
                ],
                columns=properties,
                index=index_as_ids_dtypes([0]),
            ),
            check_dtype=False,
        )

    def test_pathway_edges_1(self):
        properties = [Synapse.AXONAL_DELAY]
        pdt.assert_frame_equal(
            self.test_obj.pathway_edges([0, 1], [1, 2], properties),
            pd.DataFrame(
                [
                    [88.1862],
                    [52.1881],
                ],
                columns=properties,
                index=index_as_ids_dtypes([1, 2]),
            ),
            check_dtype=False,
        )

    def test_pathway_edges_2(self):
        tested = self.test_obj.pathway_edges([1, 2], [0, 2], None)
        npt.assert_equal(tested, [0])
        npt.assert_equal(tested.dtype, IDS_DTYPE)

    def test_pathway_edges_3(self):
        npt.assert_equal(self.test_obj.pathway_edges([0, 1], None, None), [1, 2])

    def test_pathway_edges_4(self):
        npt.assert_equal(self.test_obj.pathway_edges(None, [0, 1], None), [0, 1, 2, 3])

    def test_pathway_edges_5(self):
        with pytest.raises(BluepySnapError):
            self.test_obj.pathway_edges(None, None, None)

    def test_pathway_edges_6(self):
        ids = CircuitNodeIds.from_dict({"default": [0, 1]})
        npt.assert_equal(self.test_obj.pathway_edges(ids, None, None), [1, 2])

    def test_iter_connections_1(self):
        it = self.test_obj.iter_connections([0, 2], [1])
        assert next(it) == (
            CircuitNodeId(population="default", id=0),
            CircuitNodeId(population="default", id=1),
        )
        assert next(it) == (
            CircuitNodeId(population="default", id=2),
            CircuitNodeId(population="default", id=1),
        )
        with pytest.raises(StopIteration):
            next(it)

    def test_iter_connections_2(self):
        it = self.test_obj.iter_connections([0, 2], [1], unique_node_ids=True)
        assert list(it) == [
            (CircuitNodeId(population="default", id=0), CircuitNodeId(population="default", id=1)),
        ]

    def test_iter_connections_3(self):
        it = self.test_obj.iter_connections([0, 2], [1], shuffle=True)
        assert sorted(it) == [
            (CircuitNodeId(population="default", id=0), CircuitNodeId(population="default", id=1)),
            (CircuitNodeId(population="default", id=2), CircuitNodeId(population="default", id=1)),
        ]

    def test_iter_connections_4(self):
        it = self.test_obj.iter_connections(None, None)
        with pytest.raises(BluepySnapError):
            next(it)

    def test_iter_connections_5(self):
        it = self.test_obj.iter_connections(None, [1])
        assert list(it) == [
            (CircuitNodeId(population="default", id=0), CircuitNodeId(population="default", id=1)),
            (CircuitNodeId(population="default", id=2), CircuitNodeId(population="default", id=1)),
        ]

    def test_iter_connections_6(self):
        it = self.test_obj.iter_connections([2], None)
        assert list(it) == [
            (CircuitNodeId(population="default", id=2), CircuitNodeId(population="default", id=0)),
            (CircuitNodeId(population="default", id=2), CircuitNodeId(population="default", id=1)),
        ]

    def test_iter_connections_7(self):
        it = self.test_obj.iter_connections([], [0, 1, 2])
        assert list(it) == []

    def test_iter_connections_8(self):
        it = self.test_obj.iter_connections([0, 2], [1], return_edge_ids=True)
        npt.assert_equal(
            list(it),
            [
                (
                    CircuitNodeId(population="default", id=0),
                    CircuitNodeId(population="default", id=1),
                    CircuitEdgeIds.from_dict({"default": [1, 2]}),
                ),
                (
                    CircuitNodeId(population="default", id=2),
                    CircuitNodeId(population="default", id=1),
                    CircuitEdgeIds.from_dict({"default": [3]}),
                ),
            ],
        )

    def test_iter_connections_9(self):
        it = self.test_obj.iter_connections([0, 2], [1], return_edge_count=True)
        assert list(it) == [
            (
                CircuitNodeId(population="default", id=0),
                CircuitNodeId(population="default", id=1),
                2,
            ),
            (
                CircuitNodeId(population="default", id=2),
                CircuitNodeId(population="default", id=1),
                1,
            ),
        ]

    def test_iter_connections_10(self):
        with pytest.raises(BluepySnapError):
            self.test_obj.iter_connections(
                [0, 2], [1], return_edge_ids=True, return_edge_count=True
            )

    def test_iter_connection_unique(self):
        with copy_test_data() as (config_dir, config_path):
            with edit_config(config_path) as config:
                config["networks"]["edges"] = [
                    {
                        "edge_types_file": None,
                        "edges_file": str(Path(config_dir) / "edges_complete_graph.h5"),
                        "populations": {"default": {"type": "chemical"}},
                    }
                ]

            test_obj = TestEdgePopulation.get_edge_population(config_path, "default")

            it = test_obj.iter_connections([0, 1, 2], [0, 1, 2])
            assert sorted(it) == [
                (
                    CircuitNodeId(population="default", id=0),
                    CircuitNodeId(population="default", id=1),
                ),
                (
                    CircuitNodeId(population="default", id=0),
                    CircuitNodeId(population="default", id=2),
                ),
                (
                    CircuitNodeId(population="default", id=1),
                    CircuitNodeId(population="default", id=0),
                ),
                (
                    CircuitNodeId(population="default", id=1),
                    CircuitNodeId(population="default", id=2),
                ),
                (
                    CircuitNodeId(population="default", id=2),
                    CircuitNodeId(population="default", id=0),
                ),
                (
                    CircuitNodeId(population="default", id=2),
                    CircuitNodeId(population="default", id=1),
                ),
            ]

            it = test_obj.iter_connections([0, 1, 2], [0, 1, 2], unique_node_ids=True)
            assert sorted(it) == [
                (
                    CircuitNodeId(population="default", id=0),
                    CircuitNodeId(population="default", id=1),
                ),
                (
                    CircuitNodeId(population="default", id=1),
                    CircuitNodeId(population="default", id=0),
                ),
            ]

            it = test_obj.iter_connections([0, 1, 2], [0, 2], unique_node_ids=True)
            assert sorted(it) == [
                (
                    CircuitNodeId(population="default", id=0),
                    CircuitNodeId(population="default", id=2),
                ),
                (
                    CircuitNodeId(population="default", id=1),
                    CircuitNodeId(population="default", id=0),
                ),
            ]

            it = test_obj.iter_connections([0, 2], [0, 2], unique_node_ids=True)
            assert sorted(it) == [
                (
                    CircuitNodeId(population="default", id=0),
                    CircuitNodeId(population="default", id=2),
                ),
                (
                    CircuitNodeId(population="default", id=2),
                    CircuitNodeId(population="default", id=0),
                ),
            ]

            it = test_obj.iter_connections([0, 1, 2], [0, 2, 1], unique_node_ids=True)
            assert sorted(it) == [
                (
                    CircuitNodeId(population="default", id=0),
                    CircuitNodeId(population="default", id=1),
                ),
                (
                    CircuitNodeId(population="default", id=1),
                    CircuitNodeId(population="default", id=0),
                ),
            ]

            it = test_obj.iter_connections([1, 2], [0, 1, 2], unique_node_ids=True)
            assert sorted(it) == [
                (
                    CircuitNodeId(population="default", id=1),
                    CircuitNodeId(population="default", id=0),
                ),
                (
                    CircuitNodeId(population="default", id=2),
                    CircuitNodeId(population="default", id=1),
                ),
            ]

            it = test_obj.iter_connections([0, 1, 2], [1, 2], unique_node_ids=True)
            assert sorted(it) == [
                (
                    CircuitNodeId(population="default", id=0),
                    CircuitNodeId(population="default", id=1),
                ),
                (
                    CircuitNodeId(population="default", id=1),
                    CircuitNodeId(population="default", id=2),
                ),
            ]

    def test_h5_filepath_from_config(self):
        assert self.test_obj.h5_filepath == str(TEST_DATA_DIR / "edges.h5")

    @pytest.mark.skip(reason="Until spatial-index is released publicly")
    def test_spatial_synapse_index(self):
        with mock.patch("spatial_index.open_index") as mock_open_index:
            self.test_obj.spatial_synapse_index
        mock_open_index.assert_called_once_with("path/to/edge/dir")

    @mock.patch.dict(sys.modules, {"spatial_index": mock.Mock()})
    def test_spatial_synapse_index_call(self):
        with pytest.raises(
            BluepySnapError,
            match="It appears default does not have synapse indices",
        ):
            self.test_obj.spatial_synapse_index

    def test_spatial_synapse_index_error(self):
        with pytest.raises(
            BluepySnapError,
            match=(
                "Spatial index is for now only available internally to BBP. "
                "It requires `spatial_index`, an internal package."
            ),
        ):
            self.test_obj.spatial_synapse_index

    def test_pickle(self, tmp_path):
        pickle_path = tmp_path / "pickle.pkl"

        # trigger some cached properties, to makes sure they aren't being pickeld
        self.test_obj.source
        self.test_obj.target
        self.test_obj.property_dtypes

        with open(pickle_path, "wb") as fd:
            pickle.dump(self.test_obj, fd)

        with open(pickle_path, "rb") as fd:
            edge_population = pickle.load(fd)

        assert pickle_path.stat().st_size < 130 + PICKLED_SIZE_ADJUSTMENT
        assert edge_population.name == "default"


class TestEdgePopulationSpatialIndex:
    def setup_method(self):
        self.test_obj = TestEdgePopulation.get_edge_population(
            TEST_DATA_DIR / "circuit_config.json", "default2"
        )

    @mock.patch.dict(sys.modules, {"spatial_index": mock.Mock()})
    def test_spatial_synapse_index_call(self):
        self.test_obj.spatial_synapse_index
        mock = sys.modules["spatial_index"].open_index
        assert mock.call_count == 1
        assert mock.call_args[0][0].endswith("path/to/edge/dir")
