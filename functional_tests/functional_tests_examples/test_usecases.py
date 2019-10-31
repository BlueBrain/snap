from __future__ import print_function

import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from difflib import Differ

import six

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


@contextmanager
def setup_tempdir(prefix, cleanup=True):
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield temp_dir
    finally:
        if cleanup:
            shutil.rmtree(temp_dir)


def get_notebooks():
    """Retrieve example notebooks."""
    notebook_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 '../../doc/source/notebooks')
    notebooks = list()
    for nb_file in glob.glob(os.path.join(notebook_path, "*.ipynb")):
        notebooks.append(os.path.join(notebook_path, nb_file))
    return sorted(notebooks)


def _sanitize(s):
    """ Sanitizes string for comparison """
    # ignore trailing newlines
    s = s.rstrip('\r\n')
    # normalize hex addresses:
    s = re.sub(r'0x[A-Fa-f0-9]+', '0xFFFFFFFF', s)
    # normalize UUIDs:
    s = re.sub(r'[a-f0-9]{8}(\-[a-f0-9]{4}){3}\-[a-f0-9]{12}', 'U-U-I-D', s)
    # normalize <th> and <tr> (pandas changed this multiple times for df outputs)
    s = re.sub(r'<td|<th', '<t-', s)
    s = re.sub(r'</td>|</th>', '</t->', s)
    return s


def verifies_notebook(path, dry_run):
    """Run the test and compare actual inputs to the newly produced ones."""
    with setup_tempdir(os.path.join(TEST_DIR, "tmp")) as tmp:
        output = os.path.join(tmp, "tested.ipynb")
        subprocess.call(
            ["jupyter", "nbconvert", "--to", "notebook", "--execute", path, "--output", output])
        if not dry_run:
            ref = json.load(open(path))["cells"]
            tested = json.load(open(output))["cells"]
            for ref_cell, tested_cell in zip(ref, tested):
                ref_cell = _sanitize(json.dumps(ref_cell))
                tested_cell = _sanitize(json.dumps(tested_cell))
                diff = Differ().compare(ref_cell.splitlines(True), tested_cell.splitlines(True))
                # will show both lines
                assert len(list(diff)) == 1


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
