import os
from unittest.mock import patch

import bluepysnap.settings as test_module


def test_str2bool():
    assert test_module.str2bool("1")
    assert test_module.str2bool("y")
    assert test_module.str2bool("YES")
    assert not test_module.str2bool("0")
    assert not test_module.str2bool("n")
    assert not test_module.str2bool("No")
    assert not test_module.str2bool(None)


def test_STRICT_MODE():
    with patch.dict(os.environ, {"BLUESNAP_STRICT_MODE": "1"}):
        test_module.load_env()
    assert test_module.STRICT_MODE
