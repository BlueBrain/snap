'''Module providing utility functions for the tests'''

import shutil
import tempfile
from contextlib import contextmanager

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

TEST_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = TEST_DIR / 'data'


@contextmanager
def setup_tempdir(cleanup=True):
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        if cleanup:
            shutil.rmtree(temp_dir)
