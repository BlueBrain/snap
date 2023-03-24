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
    "object": _str_dtype,
    "str": _str_dtype,
    "float32": np.float32,
    "float": float,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int": int,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint": np.uint64,
}


def add_test_field(file_path, population_name, data, data_type):
    pop_0_path = f"/nodes/{population_name}/0"
    test_data_path = f"{pop_0_path}/{TEST_FIELD}"

    with h5py.File(file_path, "r+") as h5:
        if data_type == "category":
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
    ("dtypes", "expected"),
    (
        (("category", "category"), "category"),
        (("int8", "int8"), "int8"),
        (("uint8", "uint8"), "uint8"),
        (("object", "object"), "object"),
        (("category", "str"), "category"),
        (("category", "int"), "category"),
        (("int", "float"), "float"),
        (("int", "str"), "object"),
        (("int", "int16"), "int"),
        (("int", "int32"), "int"),
        (("uint32", "int32"), "int"),
        (("uint", "float"), "float"),
        (("float32", "float"), "float"),
        (("int8", "uint8"), "int16"),
        (("int16", "uint8"), "int16"),
        (("int16", "uint16"), "int32"),
        (("int32", "uint32"), "int"),
        (("int", "uint32"), "int"),
        (("int", "uint"), "float"),
    ),
)
def test_resulting_dtypes(dtypes, expected):
    with copy_test_data() as (test_dir, config_path):
        node_path = Path(test_dir) / "nodes.h5"
        for population, data, dtype in zip(TEST_POPULATIONS, TEST_DATA, dtypes):
            add_test_field(node_path, population, data, dtype)

        res = Circuit(config_path).nodes.get(properties=TEST_FIELD)
        assert res[TEST_FIELD].dtype == expected
