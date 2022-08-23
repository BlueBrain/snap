import sys
from unittest.mock import patch

import pytest

import bluepysnap._plotting as test_module


def test__get_pyplot():
    with patch.dict(sys.modules, {"matplotlib.pyplot": None}):
        with pytest.raises(ImportError):
            test_module._get_pyplot()

    import matplotlib.pyplot

    plt_test = test_module._get_pyplot()
    assert plt_test is matplotlib.pyplot
