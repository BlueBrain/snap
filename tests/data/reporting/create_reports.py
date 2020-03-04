import h5py
import numpy as np


def write_spikes(filepath):
    population_names = ['default', 'default2']
    timestamps_base = (0.3, 0.1, 0.2, 1.3, 0.7)
    node_ids_base = (1, 2, 0, 0, 2)

    sorting_type = h5py.enum_dtype({"none": 0, "by_id": 1, "by_time": 2})
    string_dtype = h5py.special_dtype(vlen=str)

    with h5py.File(filepath, 'w') as h5f:
        root = h5f.create_group('spikes')
        gpop_all = h5f.create_group('/spikes/' + population_names[0])
        gpop_all.attrs.create('sorting', data=2, dtype=sorting_type)
        timestamps, node_ids = zip(*sorted(zip(timestamps_base, node_ids_base)))
        set = gpop_all.create_dataset('timestamps', data=timestamps, dtype=np.double)
        gpop_all.create_dataset('node_ids', data=node_ids, dtype=np.uint64)

        gpop_spikes1 = h5f.create_group('/spikes/' + population_names[1])
        gpop_spikes1.attrs.create('sorting', data=1, dtype=sorting_type)
        node_ids, timestamps = zip(*sorted(zip(node_ids_base, timestamps_base)))
        gpop_spikes1.create_dataset('timestamps', data=timestamps, dtype=np.double)
        gpop_spikes1.create_dataset('node_ids', data=node_ids, dtype=np.uint64)


if __name__ == "__main__":
    write_spikes("spikes.h5")
