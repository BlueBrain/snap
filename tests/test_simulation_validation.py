from unittest.mock import Mock, patch

import bluepysnap.simulation_validation as test_module

from utils import TEST_DATA_DIR


@patch.object(test_module, "print_validation_errors")
def test_validate(mock_print):
    config = TEST_DATA_DIR / "simulation_config.json"

    res = test_module.validate(config, print_errors=False)
    mock_print.assert_not_called()
    assert res == set()

    res = test_module.validate(config, print_errors=True)
    mock_print.assert_called_once_with([])
    assert res == set()
