import collections
from bluepysnap import utils


def _simplify(node_set):
    node_set = dict(node_set)
    for key, value in node_set.items():
        if isinstance(value, list) and len(value) == 1:
            node_set[key] = value[0]
    return node_set


class NodeSets:
    def __init__(self, filepath):
        self.content = utils.load_json(filepath)
        self.resolved = self._resolve()

    def _resolve_set(self, resolved, set_name):
        if set_name in resolved:
            # allows to returned already computed values
            return resolved[set_name]

        set_value = self.content[set_name]
        if isinstance(set_value, collections.MutableMapping):
            return _simplify(set_value)

        #  combined node set only (list)
        res = collections.defaultdict(list)
        for sub_set_name in set_value:
            sub_res_dict = self._resolve_set(resolved, sub_set_name)
            resolved[sub_set_name] = _simplify(sub_res_dict)
            for key, value in sub_res_dict.items():
                res[key].extend(utils.ensure_list(value))
        return _simplify(res)

    def _resolve(self):
        resolved = {}
        for set_name, set_value in self.content.items():
            resolved[set_name] = self._resolve_set(resolved, set_name)
        return resolved

    def __getitem__(self, item):
        return self.resolved[item]

    def __iter__(self):
        return self.resolved.__iter__()
