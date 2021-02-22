"""Module to process search queries of nodes/edges."""
from collections.abc import Mapping
from copy import deepcopy

import numpy as np
from bluepysnap.exceptions import BluepySnapError
from bluepysnap import utils

# this constant is not part of the sonata standard
NODE_ID_KEY = "node_id"
EDGE_ID_KEY = "edge_id"
POPULATION_KEY = "population"
OR_KEY = "$or"
AND_KEY = "$and"
REGEX_KEY = "$regex"
NODE_SET_KEY = "$node_set"
VALUE_KEYS = {REGEX_KEY}
ALL_KEYS = {NODE_ID_KEY, EDGE_ID_KEY, POPULATION_KEY, OR_KEY, AND_KEY, NODE_SET_KEY} | VALUE_KEYS


# TODO: move to `libsonata` library
def _complex_query(prop, query):
    result = np.full(len(prop), True)
    for key, value in query.items():
        if key == REGEX_KEY:
            result = np.logical_and(result, prop.str.match(value + "\\Z"))
        else:
            raise BluepySnapError("Unknown query modifier: '%s'" % key)
    return result


def _positional_mask(data, ids):
    """Positional mask for the node IDs.

    Args:
        ids (None/numpy.ndarray): the ids array. If None all ids are selected.

    Examples:
        if the data set contains 5 nodes:
        _positional_mask(data, [0,2]) --> [True, False, True, False, False]
    """
    if ids is None:
        return np.full(len(data), fill_value=True)
    if isinstance(ids, int):
        ids = [ids]
    mask = np.full(len(data), fill_value=False)
    indices = data.index.get_indexer(ids)
    mask[indices[indices > -1]] = True
    return mask


def _circuit_mask(data, population_name, queries):
    """Handle the population, node ID queries."""
    populations = queries.pop(POPULATION_KEY, None)
    if populations is not None and population_name not in set(utils.ensure_list(populations)):
        ids = []
    else:
        ids = queries.pop(NODE_ID_KEY, queries.pop(EDGE_ID_KEY, None))
    return queries, _positional_mask(data, ids)


def _properties_mask(data, population_name, queries):
    """Return mask of IDs matching `props` dict."""
    unknown_props = set(queries) - set(data.columns) - ALL_KEYS
    if unknown_props:
        return np.full(len(data), fill_value=False)

    queries, mask = _circuit_mask(data, population_name, queries)
    if not mask.any():
        # Avoid fail and/or processing time if wrong population or no nodes
        return mask

    for prop, values in queries.items():
        prop = data[prop]
        if np.issubdtype(prop.dtype.type, np.floating):
            v1, v2 = values
            prop_mask = np.logical_and(prop >= v1, prop <= v2)
        elif isinstance(values, Mapping):
            prop_mask = _complex_query(prop, values)
        else:
            prop_mask = np.in1d(prop, values)
        mask = np.logical_and(mask, prop_mask)
    return mask


def traverse_queries_bottom_up(queries, traverse_fn):
    """Traverse queries tree from leaves to root, left to right.

    Args:
        queries (dict): queries
        traverse_fn (function): function to execute on each node of `queries` in traverse order
    """
    for key in list(queries.keys()):
        if key in {OR_KEY, AND_KEY}:
            for subquery in queries[key]:
                traverse_queries_bottom_up(subquery, traverse_fn)
        elif isinstance(queries[key], Mapping):
            if VALUE_KEYS & set(queries[key]):
                if not set(queries[key]).issubset(VALUE_KEYS):
                    raise BluepySnapError("Value operators can't be used with plain values")
            else:
                traverse_queries_bottom_up(queries[key], traverse_fn)
        traverse_fn(queries, key)


def get_properties(queries):
    """Extracts properties names from `queries`.

    Args:
        queries (dict): queries

    Returns:
        set: set of properties names
    """

    def _collect(_, query_key):
        if query_key not in ALL_KEYS:
            props.add(query_key)

    props = set()
    traverse_queries_bottom_up(queries, _collect)
    return props


def resolve_ids(data, population_name, queries):
    """Returns an index mask of `data` for given `queries`.

    Args:
        data (pd.DataFrame): data
        population_name (str): population name of `data`
        queries (dict): queries

    Returns:
        np.array: index mask
    """

    def _merge_queries_masks(queries):
        if len(queries) == 0:
            return np.full(len(data), True)
        return np.logical_and.reduce(list(queries.values()))

    def _collect(queries, queries_key):
        # each queries value is replaced with a bit mask of corresponding ids
        if queries_key == OR_KEY:
            # children are already resolved masks due to traverse order
            children_mask = [_merge_queries_masks(query) for query in queries[queries_key]]
            queries[queries_key] = np.logical_or.reduce(children_mask)
        elif queries_key == AND_KEY:
            # children are already resolved masks due to traverse order
            children_mask = [_merge_queries_masks(query) for query in queries[queries_key]]
            queries[queries_key] = np.logical_and.reduce(children_mask)
        else:
            queries[queries_key] = _properties_mask(
                data, population_name, {queries_key: queries[queries_key]})

    queries = deepcopy(queries)
    traverse_queries_bottom_up(queries, _collect)
    return _merge_queries_masks(queries)
