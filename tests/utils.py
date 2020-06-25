"""Module providing utility functions for the tests"""

import shutil
import tempfile
import json
import six
from contextlib import contextmanager
from distutils.dir_util import copy_tree

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


@contextmanager
def copy_circuit(config='circuit_config.json'):
    """Copies test/data circuit to a temp directory.

    We don't need the whole circuit every time but considering this is a copy into a temp dir,
    it should be fine.
    Returns:
        yields a path to the copy of the config file
    """
    with setup_tempdir() as tmp_dir:
        copy_tree(str(TEST_DATA_DIR), tmp_dir)
        circuit_copy_path = Path(tmp_dir)
        yield (circuit_copy_path, circuit_copy_path / config)


@contextmanager
def edit_config(config_path):
    """Context manager within which you can edit a circuit config. Edits are saved on the context
    manager leave.

    Args:
        config_path (Path): path to config

    Returns:
        Yields a json dict instance of the config_path. This instance will be saved as the config.
    """
    with config_path.open('r') as f:
        config = json.load(f)
    try:
        yield config
    finally:
        with config_path.open('w') as f:
            f.write(six.u(json.dumps(config)))
