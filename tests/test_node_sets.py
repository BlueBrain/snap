import json
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
                                         'alone': {'alone': True}}

        assert self.test_obj.resolved == {'Node2_L6_Y': {'mtype': 'L6_Y', 'node_id': 2},
                                          'Layer23': {'layer': [2, 3]},
                                          'combined': {'mtype': 'L6_Y', 'node_id': 2,
                                                       'layer': [2, 3]},
                                          'other': {'something': [2, 3], 'something_2': ['a', 'b'],
                                                    'mtype': 'L2_X'},
                                          'double_combined': {'mtype': ['L6_Y', 'L2_X'],
                                                              'node_id': 2, 'layer': [2, 3],
                                                              'something': [2, 3],
                                                              'something_2': ['a', 'b']},
                                          'alone': {'alone': True}}

    def test_get(self):
        assert self.test_obj['Node2_L6_Y'] == {'mtype': 'L6_Y', 'node_id': 2}
        assert self.test_obj['double_combined'] == {'mtype': ['L6_Y', 'L2_X'],
                                                    'node_id': 2, 'layer': [2, 3],
                                                    'something': [2, 3],
                                                    'something_2': ['a', 'b']}

    def test_iter(self):
        expected = sorted(list(json.load(open(str(TEST_DATA_DIR / 'node_sets_file.json')))))
        assert sorted(list(self.test_obj)) == expected
