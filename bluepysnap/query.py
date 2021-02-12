from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import pandas as pd
from bluepysnap.exceptions import BluepySnapError
from bluepysnap import utils

# this constant is not part of the sonata standard
NODE_ID_KEY = "node_id"
EDGE_ID_KEY = "edge_id"
POPULATION_KEY = "population"
OR_KEY = "$or"
AND_KEY = "$and"
NODE_SET_KEY = "$node_set"  # the only key that is not a part of QUERY_KEYS
QUERY_KEYS = {NODE_ID_KEY, EDGE_ID_KEY, POPULATION_KEY, OR_KEY, AND_KEY}


def _positional_mask(data, node_ids):
    """Positional mask for the node IDs.

    Args:
        node_ids (None/numpy.ndarray): the ids array. If None all ids are selected.

    Examples:
        if the data set contains 5 nodes:
        _positional_mask([0,2]) --> [True, False, True, False, False]
    """
    if node_ids is None:
        return np.full(len(data), fill_value=True)
    mask = np.full(len(data), fill_value=False)
    valid_node_ids = pd.Index(utils.ensure_list(node_ids)).intersection(data.index)
    mask[valid_node_ids] = True
    return mask


def _circuit_mask(data, population_name, queries):
    """Handle the population, node ID queries."""
    populations = queries.pop(POPULATION_KEY, None)
    if populations is not None and population_name not in set(utils.ensure_list(populations)):
        node_ids = []
    else:
        node_ids = queries.pop(NODE_ID_KEY, None)
    return queries, _positional_mask(data, node_ids)


def _properties_mask(data, population_name, queries):
    """Return mask of IDs matching `props` dict."""
    # pylint: disable=assignment-from-no-return
    unknown_props = set(queries) - set(data.columns) - QUERY_KEYS
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
        elif isinstance(values, str) and values.startswith('regex:'):
            prop_mask = np.logical_and(np.full(len(prop), True), prop.str.match(values[6:] + "\\Z"))
        else:
            prop_mask = np.in1d(prop, values)
        mask = np.logical_and(mask, prop_mask)
    return mask


def traverse_queries_bottom_up(queries, node_function):
    # traverse from leaves to root
    for key in queries:
        if key in QUERY_KEYS:
            if key in {OR_KEY, AND_KEY}:
                for subquery in queries[key]:
                    traverse_queries_bottom_up(subquery, node_function)
        elif isinstance(queries[key], Mapping):
            traverse_queries_bottom_up(queries[key], node_function)
        node_function(queries, key)


def get_properties(queries):
    def _collect(_, query_key):
        if query_key not in QUERY_KEYS:
            props.add(query_key)

    props = set()
    traverse_queries_bottom_up(queries, _collect)
    return props


def operator_mask(data, population_name, queries):
    def _merge_queries_masks(queries):
        if len(queries) == 0:
            return np.full(len(data), True)
        return np.logical_and.reduce([mask for mask in queries.values()])

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
