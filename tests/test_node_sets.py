import json
import pytest
from mock import patch
from bluepysnap.exceptions import BluepySnapError
import bluepysnap.node_sets as test_module

from utils import TEST_DATA_DIR


class TestNodeSets:
    def setup(self):
        self.test_obj = test_module.NodeSets(str(TEST_DATA_DIR / 'node_sets_file.json'))

    def test_init(self):
        assert self.test_obj.content == {'double_combined': ['combined', 'other'],
                                         'Node2_L6_Y': {'mtype': 'L6_Y', 'node_id': [2]},
                                         'Layer23': {'layer': [2, 3]},
                                         'other': {'something': [2, 3], 'something_2': ['a', 'b'],
                                                   'mtype': 'L2_X'},
                                         'combined': ['Node2_L6_Y', 'Layer23'],
                                         'alone': {'alone': True},
                                         'empty_dict': {}
                                         }

        assert self.test_obj.resolved == {'Node2_L6_Y': {'mtype': 'L6_Y', 'node_id': 2},
                                          'Layer23': {'layer': [2, 3]},
                                          'combined': {'mtype': 'L6_Y', 'node_id': 2,
                                                       'layer': [2, 3]},
                                          'other': {'something': [2, 3], 'something_2': ['a', 'b'],
                                                    'mtype': 'L2_X'},
                                          'double_combined': {'mtype': ['L2_X', 'L6_Y'],
                                                              'node_id': 2, 'layer': [2, 3],
                                                              'something': [2, 3],
                                                              'something_2': ['a', 'b']},
                                          'alone': {'alone': True}, 'empty_dict': {}}

    def test_get(self):
        assert self.test_obj['Node2_L6_Y'] == {'mtype': 'L6_Y', 'node_id': 2}
        assert self.test_obj['double_combined'] == {'mtype': ['L2_X', 'L6_Y'],
                                                    'node_id': 2, 'layer': [2, 3],
                                                    'something': [2, 3],
                                                    'something_2': ['a', 'b']}

    def test_iter(self):
        expected = sorted(list(json.load(open(str(TEST_DATA_DIR / 'node_sets_file.json')))))
        assert sorted(list(self.test_obj)) == expected


@patch('bluepysnap.utils.load_json')
def test_fail_resolve(mock_load):
    mock_load.return_value = {"empty_list": []}
    with pytest.raises(BluepySnapError):
        test_module.NodeSets(str(TEST_DATA_DIR / 'node_sets_file.json'))

    mock_load.return_value = {"int": 1}
    with pytest.raises(BluepySnapError):
        test_module.NodeSets(str(TEST_DATA_DIR / 'node_sets_file.json'))

    mock_load.return_value = {"bool": True}
    with pytest.raises(BluepySnapError):
        test_module.NodeSets(str(TEST_DATA_DIR / 'node_sets_file.json'))

    mock_load.return_value = {"combined": ["known", "unknown"], "known": {"v": 1}}
    with pytest.raises(BluepySnapError):
        test_module.NodeSets(str(TEST_DATA_DIR / 'node_sets_file.json'))


def test_compound():
    test_obj = test_module.NodeSets(str(TEST_DATA_DIR / 'node_sets_file_compound_population.json'))
    print(test_obj.resolved)
    assert False
