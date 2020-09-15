import sys

import mock
import pytest
import six

import bluepysnap._plotting as test_module


def test__get_pyplot():
    with mock.patch.dict(sys.modules, {'matplotlib.pyplot': None}):
        with pytest.raises(ImportError):
            test_module._get_pyplot()

    if sys.platform == 'darwin' and six.PY2:
        # Set a backend different to MacOSX (default on Mac OS) to avoid the error
        #   RuntimeError: Python is not installed as a framework
        # using Python 2.7 in a virtualenv on Mac OS.
        # See also https://matplotlib.org/faq/osx_framework.html.
        import matplotlib
        matplotlib.use('pdf')

    import matplotlib.pyplot
    plt_test = test_module._get_pyplot()
    assert plt_test is matplotlib.pyplot
