from pathlib2 import Path

import bluepysnap.circuit_validation as test_module

TEST_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = TEST_DIR / 'data'


def test_all():
    errors = test_module.validate(str(TEST_DATA_DIR / 'circuit_config.json'))
