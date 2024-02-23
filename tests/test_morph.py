import h5py
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pytest

import bluepysnap.morph as test_module
from bluepysnap.circuit import Circuit
from bluepysnap.circuit_ids_types import CircuitNodeId
from bluepysnap.exceptions import BluepySnapError
from bluepysnap.sonata_constants import Node

from utils import TEST_DATA_DIR, copy_test_data, create_node_population, edit_config


class TestMorphHelper:
    def setup_method(self):
        self.nodes = create_node_population(str(TEST_DATA_DIR / "nodes_quaternions.h5"), "default")
        self.morph_path = TEST_DATA_DIR / "morphologies"
        self.test_obj = test_module.MorphHelper(str(self.morph_path), self.nodes)

    def test_biophysical_in_library(self):
        with copy_test_data() as (circuit_copy_path, config_copy_path):
            with edit_config(config_copy_path) as config:
                config["networks"]["nodes"][0]["nodes_file"] = "$NETWORK_DIR/nodes_quaternions.h5"
            nodes_file = circuit_copy_path / "nodes_quaternions.h5"
            with h5py.File(nodes_file, "r+") as h5f:
                data = h5f["nodes/default/0/model_type"][:]
                del h5f["nodes/default/0/model_type"]
                h5f.create_dataset(
                    "nodes/default/0/model_type", data=np.zeros_like(data, dtype=int)
                )
                h5f.create_dataset(
                    "nodes/default/0/@library/model_type",
                    data=np.array(
                        [
                            "biophysical",
                        ],
                        dtype=h5py.string_dtype(),
                    ),
                )

            circuit = Circuit(str(config_copy_path))
            assert isinstance(circuit.nodes["default"].morph, test_module.MorphHelper)

    def test_get_morphology_dir(self):
        with pytest.raises(
            BluepySnapError, match="'neurolucida-asc' is not defined in 'alternate_morphologies'"
        ):
            self.test_obj.get_morphology_dir("asc")

        with pytest.raises(BluepySnapError, match="Unsupported extension: fake"):
            self.test_obj.get_morphology_dir("fake")

        assert self.test_obj.get_morphology_dir("swc") == str(TEST_DATA_DIR / "morphologies")

    def test_get_filepath(self):
        node_id = 0
        assert self.nodes.get(node_id, properties="morphology") == "morph-A"

        actual = self.test_obj.get_filepath(node_id)
        expected = self.morph_path / "morph-A.swc"
        assert actual == expected

        actual = self.test_obj.get_filepath(np.int64(node_id))
        assert actual == expected

        actual = self.test_obj.get_filepath(np.uint64(node_id))
        assert actual == expected

        actual = self.test_obj.get_filepath(np.int32(node_id))
        assert actual == expected

        actual = self.test_obj.get_filepath(np.uint32(node_id))
        assert actual == expected

        node_id = CircuitNodeId("default", 0)
        assert self.nodes.get(node_id, properties="morphology") == "morph-A"
        actual = self.test_obj.get_filepath(node_id)
        assert actual == expected

        with pytest.raises(BluepySnapError, match="node_id must be a int or a CircuitNodeId"):
            self.test_obj.get_filepath([CircuitNodeId("default", 0), CircuitNodeId("default", 1)])

        with pytest.raises(BluepySnapError, match="node_id must be a int or a CircuitNodeId"):
            self.test_obj.get_filepath([0, 1])

    def test_alternate_morphology(self):
        alternate_morphs = {"h5v1": str(self.morph_path)}
        test_obj = test_module.MorphHelper(
            None, self.nodes, alternate_morphologies=alternate_morphs
        )

        node_id = CircuitNodeId("default", 1)
        assert self.nodes.get(node_id, properties="morphology") == "morph-B"
        expected = self.morph_path / "morph-B.h5"
        actual = test_obj.get_filepath(node_id, extension="h5")
        assert actual == expected

        alternate_morphs = {"neurolucida-asc": str(self.morph_path)}
        test_obj = test_module.MorphHelper(
            None, self.nodes, alternate_morphologies=alternate_morphs
        )

        node_id = CircuitNodeId("default", 1)
        assert self.nodes.get(node_id, properties="morphology") == "morph-B"
        expected = self.morph_path / "morph-B.asc"
        actual = test_obj.get_filepath(node_id, extension="asc")
        assert actual == expected

        with pytest.raises(BluepySnapError, match="'morphologies_dir' is not defined in config"):
            node_id = CircuitNodeId("default", 0)
            test_obj.get_filepath(node_id)

    def test_get_morphology(self):
        actual = self.test_obj.get(0)
        assert len(actual.points) == 13
        expected = [
            [0.0, 5.0, 0.0],
            [2.0, 9.0, 0.0],
        ]
        npt.assert_almost_equal(expected, actual.points[:2])
        npt.assert_almost_equal([2.0, 2.0], actual.diameters[:2])

        with pytest.raises(BluepySnapError, match="node_id must be a int or a CircuitNodeId"):
            self.test_obj.get([0, 1])

    def test_get_alternate_morphology(self):
        alternate_morphs = {"h5v1": str(self.morph_path)}
        test_obj = test_module.MorphHelper(
            None, self.nodes, alternate_morphologies=alternate_morphs
        )
        actual = test_obj.get(0, extension="h5")
        assert len(actual.points) == 13
        expected = [
            [0.0, 5.0, 0.0],
            [2.0, 9.0, 0.0],
        ]
        npt.assert_almost_equal(expected, actual.points[:2])
        npt.assert_almost_equal([2.0, 2.0], actual.diameters[:2])

    def test_get_morphology_simple_rotation(self):
        node_id = 0
        # check that the input node positions / orientation values are still the same
        pdt.assert_series_equal(
            self.nodes.positions(node_id),
            pd.Series([101.0, 102.0, 103.0], index=[Node.X, Node.Y, Node.Z], name=0),
        )
        npt.assert_almost_equal(
            self.nodes.orientations(node_id),
            [
                [1, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            decimal=6,
        )

        actual = self.test_obj.get(node_id, transform=True)
        assert len(actual.points) == 13
        # swc file
        # index       type         X            Y            Z       radius       parent
        #   22           2     0.000000     5.000000     0.000000     1.000000           1
        #   23           2     2.000000     9.000000     0.000000     1.000000          22
        # rotation around the x axis 90 degrees counter clockwise (swap Y and Z)
        # x = X + 101, y = Z + 102, z = Y + 103, radius does not change
        expected = [[101.0, 102.0, 108.0], [103.0, 102.0, 112.0]]
        npt.assert_almost_equal(actual.points[:2], expected)
        npt.assert_almost_equal(actual.diameters[:2], [2.0, 2.0])

    def test_get_morphology_standard_rotation(self):
        nodes = create_node_population(str(TEST_DATA_DIR / "nodes.h5"), "default")
        test_obj = test_module.MorphHelper(str(self.morph_path), nodes)

        node_id = 0
        actual = test_obj.get(node_id, transform=True).points

        # check if the input node positions / orientation values are still the same
        pdt.assert_series_equal(
            self.nodes.positions(node_id),
            pd.Series([101.0, 102.0, 103.0], index=[Node.X, Node.Y, Node.Z], name=0),
        )
        npt.assert_almost_equal(
            nodes.orientations(node_id),
            [
                [0.738219, 0.0, 0.674560],
                [0.0, 1.0, 0.0],
                [-0.674560, 0.0, 0.738219],
            ],
            decimal=6,
        )

        assert len(actual) == 13
        expected = [[101.0, 107.0, 103.0], [102.47644, 111.0, 101.65088]]
        npt.assert_almost_equal(actual[:2], expected, decimal=6)
