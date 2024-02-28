"""Taken from the libsonata lib."""

import h5py
import numpy as np


def write_spikes(filepath):
    population_names = ["default", "default2"]
    timestamps_base = (0.3, 0.1, 0.2, 1.3, 0.7)
    node_ids_base = (1, 2, 0, 0, 2)

    sorting_type = h5py.enum_dtype({"none": 0, "by_id": 1, "by_time": 2})

    with h5py.File(filepath, "w") as h5f:
        h5f.create_group("spikes")
        gpop_spikes = h5f.create_group("/spikes/" + population_names[0])
        gpop_spikes.attrs.create("sorting", data=2, dtype=sorting_type)
        timestamps, node_ids = zip(*sorted(zip(timestamps_base, node_ids_base)))
        gpop_spikes.create_dataset("timestamps", data=timestamps, dtype=np.double)
        gpop_spikes.create_dataset("node_ids", data=node_ids, dtype=np.uint64)

        gpop_spikes2 = h5f.create_group("/spikes/" + population_names[1])
        gpop_spikes2.attrs.create("sorting", data=1, dtype=sorting_type)
        node_ids, timestamps = zip(*sorted(zip(node_ids_base, timestamps_base)))
        gpop_spikes2.create_dataset("timestamps", data=timestamps, dtype=np.double)
        gpop_spikes2.create_dataset("node_ids", data=node_ids, dtype=np.uint64)


def write_soma_report(filepath):
    population_names = ["default", "default2"]
    node_ids = np.arange(0, 3)
    index_pointers = np.arange(0, 4)
    element_ids = np.zeros(3)
    times = (0.0, 1.0, 0.1)
    data = [node_ids + j * 0.1 for j in range(10)]
    string_dtype = h5py.special_dtype(vlen=str)
    with h5py.File(filepath, "w") as h5f:
        h5f.create_group("report")
        gpop_all = h5f.create_group("/report/" + population_names[0])
        ddata = gpop_all.create_dataset("data", data=data, dtype=np.float32)
        ddata.attrs.create("units", data="mV", dtype=string_dtype)
        gmapping = h5f.create_group("/report/" + population_names[0] + "/mapping")

        dnodes = gmapping.create_dataset("node_ids", data=node_ids, dtype=np.uint64)
        dnodes.attrs.create("sorted", data=True, dtype=np.uint8)
        gmapping.create_dataset("index_pointers", data=index_pointers, dtype=np.uint64)
        gmapping.create_dataset("element_ids", data=element_ids, dtype=np.uint32)
        dtimes = gmapping.create_dataset("time", data=times, dtype=np.double)
        dtimes.attrs.create("units", data="ms", dtype=string_dtype)

        gpop_soma2 = h5f.create_group("/report/" + population_names[1])
        ddata = gpop_soma2.create_dataset("data", data=data, dtype=np.float32)
        ddata.attrs.create("units", data="mV", dtype=string_dtype)
        gmapping = h5f.create_group("/report/" + population_names[1] + "/mapping")

        dnodes = gmapping.create_dataset("node_ids", data=node_ids, dtype=np.uint64)
        dnodes.attrs.create("sorted", data=True, dtype=np.uint8)
        gmapping.create_dataset("index_pointers", data=index_pointers, dtype=np.uint64)
        gmapping.create_dataset("element_ids", data=element_ids, dtype=np.uint32)
        dtimes = gmapping.create_dataset("time", data=times, dtype=np.double)
        dtimes.attrs.create("units", data="ms", dtype=string_dtype)


def write_element_report(filepath):
    population_names = ["default", "default2"]
    node_ids = np.arange(0, 3)
    index_pointers = np.arange(0, 8, 2)
    index_pointers[-1] = index_pointers[-1] + 1
    element_ids = np.array([0, 1] * 3 + [1])

    times = (0.0, 1, 0.1)

    string_dtype = h5py.special_dtype(vlen=str)
    with h5py.File(filepath, "w") as h5f:
        h5f.create_group("report")
        gpop_element = h5f.create_group("/report/" + population_names[0])
        d1 = np.array([np.arange(7) + j * 0.1 for j in range(10)])
        ddata = gpop_element.create_dataset("data", data=d1, dtype=np.float32)
        ddata.attrs.create("units", data="mV", dtype=string_dtype)
        gmapping = h5f.create_group("/report/" + population_names[0] + "/mapping")

        dnodes = gmapping.create_dataset("node_ids", data=node_ids, dtype=np.uint64)
        dnodes.attrs.create("sorted", data=True, dtype=np.uint8)
        gmapping.create_dataset("index_pointers", data=index_pointers, dtype=np.uint64)
        gmapping.create_dataset("element_ids", data=element_ids, dtype=np.uint32)
        dtimes = gmapping.create_dataset("time", data=times, dtype=np.double)
        dtimes.attrs.create("units", data="ms", dtype=string_dtype)

        gpop_element2 = h5f.create_group("/report/" + population_names[1])
        d1 = np.array([np.arange(7) + j * 0.1 for j in range(10)])
        ddata = gpop_element2.create_dataset("data", data=d1, dtype=np.float32)
        ddata.attrs.create("units", data="mR", dtype=string_dtype)
        gmapping = h5f.create_group("/report/" + population_names[1] + "/mapping")

        dnodes = gmapping.create_dataset("node_ids", data=node_ids, dtype=np.uint64)
        dnodes.attrs.create("sorted", data=True, dtype=np.uint8)
        gmapping.create_dataset("index_pointers", data=index_pointers, dtype=np.uint64)
        gmapping.create_dataset("element_ids", data=element_ids, dtype=np.uint32)
        dtimes = gmapping.create_dataset("time", data=times, dtype=np.double)
        dtimes.attrs.create("units", data="mR", dtype=string_dtype)


if __name__ == "__main__":
    write_spikes("spikes.h5")
    write_soma_report("soma_report.h5")
    write_element_report("compartment_named.h5")
