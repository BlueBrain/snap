import os

import bluepysnap.config as test_module


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, 'data')


def test_parse():
    actual = test_module.Config.parse(
        os.path.join(TEST_DATA_DIR, 'circuit_config.json')
    )

    # check double resolution and '.' works: $COMPONENT_DIR -> $BASE_DIR -> '.'
    assert (
        actual['components']['morphologies_dir'] ==
        os.path.join(TEST_DATA_DIR, 'morphologies')
    )

    # check resolution and './' works: $NETWORK_DIR -> './'
    assert (
        actual['networks']['nodes'][0]['nodes_file'] ==
        os.path.join(TEST_DATA_DIR, 'nodes.h5')
    )
