import sys

import mock
import pytest

import bluepysnap._plotting as test_module


def test__get_pyplot():
    with mock.patch.dict(sys.modules, {'matplotlib.pyplot': None}):
        with pytest.raises(ImportError):
            test_module._get_pyplot()

    import matplotlib

    # Set a backend different to MacOSX (default on Mac OS) to avoid the error
    #   RuntimeError: Python is not installed as a framework
    # using Python 2.7 in a virtualenv on Mac OS.
    # See also https://matplotlib.org/faq/osx_framework.html.
    matplotlib.use('pdf')

    import matplotlib.pyplot
    plt_test = test_module._get_pyplot()
    assert plt_test is matplotlib.pyplot
