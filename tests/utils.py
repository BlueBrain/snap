'''Module providing utility functions for the tests'''

import shutil
import tempfile
from contextlib import contextmanager


@contextmanager
def setup_tempdir():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)
