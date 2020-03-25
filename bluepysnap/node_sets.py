import collections

import numpy as np
from bluepysnap.exceptions import BluepySnapError
from bluepysnap import utils


def _sanitize(node_set):
    for key, values in node_set.items():
        if isinstance(values, list):
            if len(values) == 1:
                node_set[key] = values[0]
            else:
                # sorted unique value list
                node_set[key] = np.unique(np.asarray(values)).tolist()
    return node_set


class NodeSets:
    def __init__(self, filepath):
        self.content = utils.load_json(filepath)
        self.resolved = self._resolve()

    def _resolve_set(self, resolved, set_name):
        if set_name in resolved:
            # return already resolved node_sets
            return resolved[set_name]

        set_value = self.content.get(set_name)
        if set_value is None:
            raise BluepySnapError("Missing node_set: '{}'".format(set_name))
        elif not isinstance(set_value, (collections.Mapping, list)) or not set_value:
            raise BluepySnapError("Ambiguous node_set: '{}'".format({set_name: set_value}))
        else:
            # keep the self.content intact
            set_value = set_value.copy()

        if isinstance(set_value, collections.Mapping):
            return _sanitize(set_value)

        res = []
        for sub_set_name in set_value:
            sub_res_dict = self._resolve_set(resolved, sub_set_name)
            resolved[sub_set_name] = sub_res_dict
            res.append(sub_res_dict)
        return {"$or": res}

    def _resolve(self):
        resolved = {}
        for set_name in self.content:
            resolved[set_name] = self._resolve_set(resolved, set_name)
        return resolved

    def __getitem__(self, item):
        return self.resolved[item]

    def __iter__(self):
        return iter(self.resolved)
