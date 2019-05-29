import os

import bluesnap.config as test_module


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, 'data')


def test_parse():
    actual = test_module.Config.parse(
        os.path.join(TEST_DATA_DIR, 'circuit_config.json')
    )
    assert (
        actual['components']['morphologies_dir'] ==
        os.path.join(TEST_DATA_DIR, 'morphologies')
    )
