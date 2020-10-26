from mock import patch
import pandas as pd
import pandas.testing as pdt
import numpy.testing as npt
import numpy as np

import bluepysnap.circuit_ids as test_module
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

from utils import setup_tempdir


def _create_index(populations, ids):
    index = pd.MultiIndex.from_arrays([populations, ids])
    index.names = ['population', 'node_ids']
    return index


def circuit_node_ids():
    return pd.MultiIndex.from_arrays([['a', 'a', 'b', 'a'], [0, 1, 0, 2]])


class TestCircuitNodeIds:
    def setup(self):
        self.test_obj = test_module.CircuitNodeIds(circuit_node_ids())

    def test_create_global_ids(self):
        tested = test_module.CircuitNodeIds.create_global_ids('a', 0)
        pdt.assert_index_equal(tested.index, _create_index(['a'], [0]))

        tested = test_module.CircuitNodeIds.create_global_ids('a', [0, 1])
        pdt.assert_index_equal(tested.index, _create_index(['a', 'a'], [0, 1]))

        # duplicate ids if id is single int
        tested = test_module.CircuitNodeIds.create_global_ids(['a', 'b'], 0)
        pdt.assert_index_equal(tested.index, _create_index(['a', 'b'], [0, 0]))

        tested = test_module.CircuitNodeIds.create_global_ids(['a', 'b'], [0, 1])
        pdt.assert_index_equal(tested.index, _create_index(['a', 'b'], [0, 1]))

        # keep ids ordering
        tested = test_module.CircuitNodeIds.create_global_ids(['a', 'b'], [1, 0])
        pdt.assert_index_equal(tested.index, _create_index(['a', 'b'], [1, 0]))

        # keep population ordering
        tested = test_module.CircuitNodeIds.create_global_ids(['b', 'a'], [0, 1])
        pdt.assert_index_equal(tested.index, _create_index(['b', 'a'], [0, 1]))

        # keep duplicates
        tested = test_module.CircuitNodeIds.create_global_ids(['a', 'a'], [0, 0])
        pdt.assert_index_equal(tested.index, _create_index(['a', 'a'], [0, 0]))
        assert tested.index.size == 2

    def test_copy(self):
        tested = self.test_obj.copy()
        assert self.test_obj is not tested
        assert self.test_obj.index is not tested.index
        assert self.test_obj == tested

    def test_len(self):
        assert len(self.test_obj) == len(circuit_node_ids())

    def test__locate(self):
        tested = self.test_obj._locate('a')
        npt.assert_equal(tested, [0, 1, 3])

        tested = self.test_obj._locate('b')
        npt.assert_equal(tested, [2])

    def test_filter_population(self):
        tested = self.test_obj.filter_population('a')
        pdt.assert_index_equal(tested.index, _create_index(['a', 'a', 'a'], [0, 1, 2]))

        tested = self.test_obj.filter_population('b')
        pdt.assert_index_equal(tested.index, _create_index(['b'], [0]))

        tested = self.test_obj.copy()
        tested.filter_population('b', inplace=True)
        pdt.assert_index_equal(tested.index, _create_index(['b'], [0]))

    def test_get_populations(self):
        tested = self.test_obj.get_populations()
        npt.assert_equal(tested, ['a', 'a', 'b', 'a'])

        tested = self.test_obj.get_populations(unique=True)
        npt.assert_equal(tested, ['a', 'b'])

    def test_get_ids(self):
        tested = self.test_obj.get_ids()
        npt.assert_equal(tested, [0, 1, 0, 2])

        tested = self.test_obj.get_ids(unique=True)
        npt.assert_equal(tested, [0, 1, 2])

    def test_append(self):
        other = test_module.CircuitNodeIds(pd.MultiIndex.from_arrays([['c', 'b', 'c'], [0, 5, 1]]))
        expected = test_module.CircuitNodeIds(_create_index(['a', 'a', 'b', 'a', 'c', 'b', 'c'],
                                                            [0, 1, 0, 2, 0, 5, 1]))
        assert self.test_obj.append(other, inplace=False) == expected

        other = test_module.CircuitNodeIds(pd.MultiIndex.from_arrays([['a'], [0]]))
        expected = test_module.CircuitNodeIds(_create_index(['a', 'a', 'b', 'a', 'a'],
                                                            [0, 1, 0, 2, 0]))
        assert self.test_obj.append(other, inplace=False) == expected

        test_obj = test_module.CircuitNodeIds(circuit_node_ids())
        other = test_module.CircuitNodeIds(pd.MultiIndex.from_arrays([['c', 'b', 'c'], [0, 5, 1]]))
        test_obj.append(other, inplace=True)
        expected = test_module.CircuitNodeIds(_create_index(['a', 'a', 'b', 'a', 'c', 'b', 'c'],
                                                            [0, 1, 0, 2, 0, 5, 1]))
        assert test_obj == expected

    def test_sample(self):
        with patch("numpy.random.choice", return_value=np.array([0, 3])):
            tested = self.test_obj.sample(2, inplace=False)
            assert tested == test_module.CircuitNodeIds(_create_index(['a', 'a'], [0, 2]))

        tested = self.test_obj.sample(2, inplace=False)
        assert len(tested) == 2

        tested = self.test_obj.sample(25, inplace=False)
        assert len(tested) == len(circuit_node_ids())

        values = circuit_node_ids()
        test_obj = test_module.CircuitNodeIds(values)
        assert len(test_obj) == 4
        test_obj.sample(1, inplace=True)
        assert len(test_obj) == 1

    def test_limit(self):
        tested = self.test_obj.limit(2, inplace=False)
        assert len(tested) == 2
        assert tested == test_module.CircuitNodeIds(_create_index(['a', 'a'], [0, 1]))

    def test_combined_operators(self):
        tested = self.test_obj.copy()
        tested.filter_population("a", inplace=True)
        tested.limit(3, inplace=True)
        assert tested == test_module.CircuitNodeIds(_create_index(['a', 'a', 'a'], [0, 1, 2]))

        tested = self.test_obj.copy()
        tested = tested.filter_population("a").limit(3)
        assert tested == test_module.CircuitNodeIds(_create_index(['a', 'a', 'a'], [0, 1, 2]))

    def test_equal(self):
        values = circuit_node_ids()
        test_obj = test_module.CircuitNodeIds(values)
        other = test_module.CircuitNodeIds(values)
        assert test_obj == other

        diff_values = pd.MultiIndex.from_arrays([['a', 'a', 'b'], [0, 1, 0]])
        other = test_module.CircuitNodeIds(diff_values)
        assert test_obj != other

        # different object
        assert test_obj != 1

    def test_printing(self):
        tested = self.test_obj.__repr__()
        expected = """CircuitNodeIds([('a', 0),
            ('a', 1),
            ('b', 0),
            ('a', 2)],
           names=['population', 'node_ids'])"""
        assert tested == expected
        assert self.test_obj.__repr__() == self.test_obj.__str__()

    def test_roundtrip(self):
        with setup_tempdir() as tmp_dir:
            output = Path(tmp_dir, "output.csv")
            self.test_obj.to_csv(str(output))
            new = test_module.CircuitNodeIds.from_csv(str(output))
            assert self.test_obj == new
