import warnings

import pytest

from bluepysnap.exceptions import BluepySnapDeprecationWarning


def test_deprecation():
    with pytest.warns(
        BluepySnapDeprecationWarning,
        match=(
            "'bluepysnap.bbp' is deprecated and will be removed in future versions. "
            f"Please use 'bluepysnap.sonata_constants' instead."
        ),
    ):
        from bluepysnap.bbp import Cell
