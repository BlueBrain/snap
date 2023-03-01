from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from bluepysnap import Circuit
from bluepysnap.exceptions import BluepySnapError

from utils import copy_test_data

TEST_FIELD = "test_field"
TEST_DATA = list("111"), list("2222")
TEST_POPULATIONS = "default", "default2"

_str_dtype = h5py.string_dtype(encoding="utf-8")

MAP_DTYPE = {
    "float": float,
    "int": int,
    "uint": np.uint64,
    "object": _str_dtype,
    "str": _str_dtype,
    "float32": np.float32,
    "int16": np.int16,
    "int32": np.int32,
    "uint32": np.uint32,
}


def add_test_field(file_path, population_name, data, data_type):
    pop_0_path = f"/nodes/{population_name}/0"
    test_data_path = f"{pop_0_path}/{TEST_FIELD}"

    with h5py.File(file_path, "r+") as h5:
        if data_type == "categorical":
            categorical = pd.Categorical(data)
            categories = categorical.categories.values

            lib_path = f"{pop_0_path}/@library/{TEST_FIELD}"

            h5.create_dataset(lib_path, data=categories)

            data = categorical.codes
            dtype = data.dtype
        else:
            dtype = MAP_DTYPE[data_type]

        h5.create_dataset(test_data_path, data=data, dtype=dtype)


@pytest.mark.parametrize(
    "dtypes",
    (
        ("categorical", "categorical"),
        ("int", "float"),
        ("int", "uint"),
        ("int", "str"),
        ("int", "int16"),
        ("int", "int32"),
        ("int16", "int32"),
        ("uint32", "int32"),
        ("uint", "float"),
    ),
)
def test_mismatching_dtypes(dtypes):
    with copy_test_data() as (test_dir, config_path):
        node_path = Path(test_dir) / "nodes.h5"
        for population, data, dtype in zip(TEST_POPULATIONS, TEST_DATA, dtypes):
            add_test_field(node_path, population, data, dtype)

        with pytest.raises(BluepySnapError, match="Same property with different dtype."):
            Circuit(config_path).nodes.property_dtypes


@pytest.mark.parametrize("dtype", list(MAP_DTYPE))
def test_matching_dtypes(dtype):
    with copy_test_data() as (test_dir, config_path):
        node_path = Path(test_dir) / "nodes.h5"
        for population, data in zip(TEST_POPULATIONS, TEST_DATA):
            add_test_field(node_path, population, data, dtype)

        res = Circuit(config_path).nodes.property_dtypes
        assert isinstance(res, pd.Series)
