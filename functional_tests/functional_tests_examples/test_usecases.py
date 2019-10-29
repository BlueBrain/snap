from __future__ import print_function
import os
import glob
import six

from .ipynb_tester import IpynbTester


def get_notebooks():
    """Retrieve example notebooks."""
    # examples are produced using python3 and matplotlib3
    notebook_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 '../../doc/source/notebooks')
    notebooks = list()
    for nb_file in glob.glob(os.path.join(notebook_path, "*.ipynb")):
        notebooks.append(os.path.join(notebook_path, nb_file))
    return sorted(notebooks)


def verifies_notebook(path, dry_run=False):
    """Run the test and compare actual inputs to the newly produced ones."""
    tester = IpynbTester(cell_timeout=-1, query_message_timeout=2)
    tester.test_notebook(path, dry_run=dry_run)


def test_usecases():
    """Single test to test all notebooks."""
    # Matplotlib >=3 does not support python2. Due to inconsistencies between
    # matplotlib2/3 outputs, we decided to perform dry runs (i.e.: checking that all
    # cells run) for python2
    dry_run = six.PY2
    notebooks = get_notebooks()
    for count, notebook in enumerate(notebooks, 1):
        print('Testing : {}'.format(notebook))
        verifies_notebook(notebook, dry_run)
        print('{}/{} OK\n'.format(count, len(notebooks)))
