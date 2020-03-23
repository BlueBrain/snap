import collections
import numpy as np
from bluepysnap.exceptions import BluepySnapError
from bluepysnap import utils
from bluepysnap.sonata_constants import POPULATION_KEY, NODE_ID_KEY

POPULATION_NODES_KEY = "population_nodes"


def _sanitize(node_set):
    node_set = dict(node_set)
    for key, values in node_set.items():
        if key == POPULATION_NODES_KEY:
            continue
        if isinstance(values, list):
            if len(values) == 1:
                node_set[key] = values[0]
            else:
                # sorted unique value list
                node_set[key] = np.unique(np.asarray(values)).tolist()

    if POPULATION_NODES_KEY in node_set:
        # combined [(pop, [1,2,3]) (pop, [7,8])] --> [(pop, [1,2,3,7,8])]
        # combined [(pop1, [1,2]), (pop2, [3,4])] if "node_ids": [7,8] -->
        #      [(pop1, [1,2,7,8]), (pop2, [3,4,7,8])]
        d = collections.defaultdict(set)
        extra_node_ids = node_set.pop(NODE_ID_KEY, [])
        for popname, node_ids in node_set[POPULATION_NODES_KEY]:
            d[popname].update(node_ids)
        node_set[POPULATION_NODES_KEY] = [
            (popname, np.unique(np.asarray(list(node_ids) + extra_node_ids)).tolist()) for
            popname, node_ids in d.items()]
    return node_set


class NodeSets:
    def __init__(self, filepath):
        self.content = utils.load_json(filepath)
        self.resolved = self._resolve()

    def _resolve_set(self, resolved, set_name):
        if set_name in resolved:
            # allows to returned already resolved node_sets
            return resolved[set_name]

        try:
            set_value = self.content[set_name]
        except KeyError:
            raise BluepySnapError("Unknown node_set: '{}'".format(set_name))

        if isinstance(set_value, collections.MutableMapping):
            # need to combine (population, id) when present at the same time (for compound)
            if POPULATION_KEY in set_value and NODE_ID_KEY in set_value:
                set_value[POPULATION_NODES_KEY] = [(
                    set_value.pop(POPULATION_KEY), set_value.pop(NODE_ID_KEY))]
            return set_value

        #  combined node set only (list)
        res = collections.defaultdict(list)
        for sub_set_name in set_value:
            sub_res_dict = self._resolve_set(resolved, sub_set_name)
            resolved[sub_set_name] = _sanitize(sub_res_dict)
            for key, value in sub_res_dict.items():
                # case where 2 node sets have the same key
                res[key].extend(utils.ensure_list(value))

        return _sanitize(res)

    def _resolve(self):
        resolved = {}
        for set_name in self.content:
            resolved[set_name] = _sanitize(self._resolve_set(resolved, set_name))
        return resolved

    def __getitem__(self, item):
        return self.resolved[item]

    def __iter__(self):
        return iter(self.resolved)
