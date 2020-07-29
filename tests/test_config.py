import bluepysnap.config as test_module
from utils import TEST_DATA_DIR, edit_config


def test_parse():
    actual = test_module.Config.parse(
        str(TEST_DATA_DIR / 'circuit_config.json')
    )

    # check double resolution and '.' works: $COMPONENT_DIR -> $BASE_DIR -> '.'
    assert (
        actual['components']['morphologies_dir'] ==
        str(TEST_DATA_DIR / 'morphologies')
    )

    # check resolution and './' works: $NETWORK_DIR -> './'
    assert (
        actual['networks']['nodes'][0]['nodes_file'] ==
        str(TEST_DATA_DIR / 'nodes.h5')
    )
