"""Standalone module that validates Sonata simulation. See ``validate-simulation`` function."""

import contextlib
import io
import os
from pathlib import Path

import h5py
import libsonata
import numpy as np
import pandas as pd

from bluepysnap import schemas
from bluepysnap.circuit_ids import CircuitNodeIds
from bluepysnap.config import Parser
from bluepysnap.exceptions import BluepySnapValidationError
from bluepysnap.node_sets import NodeSets
from bluepysnap.utils import load_json, print_validation_errors

try:
    NEURODAMUS_PRESENT = True
    import neurodamus
except ImportError:
    NEURODAMUS_PRESENT = False


def _resolve_path(path, config):
    path = Path(path)

    if path.is_absolute():
        return path

    return config["_config_dir"] / path


def _parse_config(path):
    """Parse simulation / circuit config file."""
    return Parser.parse(load_json(path), str(Path(path).parent))


def _file_exists(path):
    return Path(path).is_file() if path is not None else False


def _get_node_sets(config):
    """Get simulation node sets instance.

    Returns circuit's node sets instance extended with that of the simulation.
    If neither of them is defined (or can't be found), returns instance with no defined node sets.
    """
    node_sets = NodeSets.from_dict({})

    if _file_exists(circuit_path := config.get("network")):
        circuit_config = _parse_config(circuit_path)

        if _file_exists(circuit_node_sets_path := circuit_config.get("node_sets_file")):
            node_sets = NodeSets.from_file(circuit_node_sets_path)

    if _file_exists(simulation_node_sets_path := config.get("node_sets_file")):
        node_sets.update(NodeSets.from_file(simulation_node_sets_path))

    return node_sets


def _get_output_dir(config):
    """Resolve output directory from the config."""
    output_dir = config.get("output", {}).get("output_dir", "output")
    return _resolve_path(output_dir, config)


def _get_circuit_path(config):
    """Resolve circuit config path from the config."""
    if path := config.get("network", "circuit_config.json"):
        return _resolve_path(path, config)

    return ""


def _add_validation_parameters(config, config_path):
    """Add helper parameters to the config."""
    config["_config_dir"] = Path(config_path).parent.absolute()
    config["_output_dir"] = _get_output_dir(config)
    config["_circuit_config"] = _get_circuit_path(config)
    config["_node_sets_instance"] = _get_node_sets(config)

    return config


@contextlib.contextmanager
def _silent_neurodamus():
    """Yield a no-log, no-output (unless error) NeurodamusCore instance."""
    # No need to init MPI since we're not running anything. This also supresses some log output.
    os.environ["NEURON_INIT_MPI"] = "0"

    # Log errors only, don't save log to a file
    log_level_error_only = neurodamus.core.configuration.LogLevel.ERROR_ONLY
    neurodamus.core.configuration.GlobalConfig.verbosity = log_level_error_only
    neurodamus.core.NeurodamusCore.init(log_filename=os.devnull)

    # Suppress any console output (e.g., from Neuron)
    with contextlib.redirect_stdout(io.StringIO()):
        with contextlib.redirect_stderr(io.StringIO()):
            yield neurodamus.core.NeurodamusCore


def _warn_on_no_neurodamus(prefix):
    """A little helper to have consistent warning when neurodamus is not found."""
    message = f"{prefix}: Can not validate: Neurodamus not found in environment"
    return [BluepySnapValidationError.warning(message)]


def _validate_file_exists(path, prefix, fatal=True):
    """Validates the existence of a file.

    Note: Error is the same if ``path`` is a directory.
    """
    error = BluepySnapValidationError.fatal if fatal else BluepySnapValidationError.warning
    return [] if _file_exists(path) else [error(f"{prefix}: No such file: {path}")]


def _validate_node_set_exists(config, node_set, prefix):
    """Validates the existence of a node set."""
    node_sets = config["_node_sets_instance"]
    if node_set not in node_sets:
        return [BluepySnapValidationError.fatal(f"{prefix}: Unknown node set: '{node_set}'")]

    return []


def _validate_mechanism_variables(suffix, mechanism):
    """Check that mechanism name (=suffix) and corresponding variables are defined in neurodamus."""
    prefix = f"conditions.mechanisms.{suffix}: neurodamus"

    with _silent_neurodamus() as nd:
        nd_attrs = dir(nd.h)

    if suffix not in nd_attrs:
        return [BluepySnapValidationError.fatal(f"{prefix}: Unknown SUFFIX: {suffix}")]

    errors = []
    for variable in mechanism:
        if (suffixed_var := f"{variable}_{suffix}") not in nd_attrs:
            message = f"{prefix}: Unknown variable: {suffixed_var}"
            errors += [BluepySnapValidationError.fatal(message)]

    return errors


def _validate_mechanisms(mechanisms):
    """Validate the 'conditions.mechanisms' section in the config."""
    if NEURODAMUS_PRESENT:
        errors = []
        for name, mechanism in mechanisms.items():
            errors += _validate_mechanism_variables(name, mechanism)

        return errors

    return _warn_on_no_neurodamus("conditions.mechanisms")


def validate_conditions(config):
    """Validate the 'conditions' section in the config."""
    conditions = config.get("conditions", {})
    modifications = conditions.get("modifications", {})
    errors = []

    for mod_name, mod_config in modifications.items():
        if "node_set" in mod_config:
            prefix = f"conditions.modifications.{mod_name}.node_set"
            errors += _validate_node_set_exists(config, mod_config["node_set"], prefix=prefix)

    if (mechanisms := conditions.get("mechanisms")) is not None:
        errors += _validate_mechanisms(mechanisms)

    return errors


def _validate_mod_override(mod_override, prefix):
    if NEURODAMUS_PRESENT:
        with _silent_neurodamus() as nd:
            try:
                nd.load_hoc(f"{mod_override}Helper")
            except RuntimeError as e:
                return [BluepySnapValidationError.fatal(f"{prefix}: neurodamus: {e.args[0]}")]

        return []

    return _warn_on_no_neurodamus(prefix)


def _validate_override(idx, item, config):
    """Helper function to validate a single connection override."""
    errors = []
    prefix = f"connection_overrides[{idx}]"

    if (key := "source") in item:
        errors += _validate_node_set_exists(config, item[key], prefix=f"{prefix}.{key}")
    if (key := "target") in item:
        errors += _validate_node_set_exists(config, item[key], prefix=f"{prefix}.{key}")
    if (key := "modoverride") in item:
        errors += _validate_mod_override(item[key], prefix=f"{prefix}.{key}")

    return errors


def validate_connection_overrides(config):
    """Validate the 'connection_overrides' section in the config."""
    overrides = config.get("connection_overrides")
    errors = []

    if isinstance(overrides, list):
        for idx, override in enumerate(overrides):
            errors += _validate_override(idx, override, config)

    return errors


def _get_ids_from_spike_file(file_):
    """Get unique gids from an input spikes file."""
    file_ = Path(file_)
    suffix = file_.suffix
    if suffix == ".dat":
        spikes = pd.read_csv(file_, delimiter=r"\s+", skiprows=1, header=None, names=["t", "id"])
        return set(spikes["id"].values - 1)
    elif suffix == ".h5":
        spikes = libsonata.SpikeReader(file_)
        populations = spikes.get_population_names()
        return {pop: set(spikes[pop].get_dict()["node_ids"]) for pop in populations}

    raise IOError(f"Unknown file type: '{suffix}' (supported: '.h5', '.dat')")


def _get_ids_from_node_set(node_set, config):
    """Get unique gids per population from a node set."""
    circuit = libsonata.CircuitConfig.from_file(config["_circuit_config"])
    node_set = config["_node_sets_instance"][node_set]
    populations = [circuit.node_population(p) for p in circuit.node_populations]
    ids_per_population = {}

    for population in populations:
        if len(ids := node_set.get_ids(population, raise_missing_property=False)):
            ids_per_population[population.name] = ids

    return ids_per_population


def _get_missing_ids(sub_ids, super_ids):
    """Get `sub_ids` ids missing from `super_ids`."""
    if isinstance(sub_ids, set):
        # if sub_ids is a set, any population having such id suffices
        super_ids = set(np.concatenate([*super_ids.values()]))
        return list(sub_ids - super_ids)

    # list conversion required by CircuitNodeIds.from_dict
    missing_per_pop = {p: list(set(sub_ids[p]) - set(super_ids.get(p, []))) for p in sub_ids}

    return CircuitNodeIds.from_dict(missing_per_pop).tolist()


def _compare_ids(sub_ids, super_ids, super_source, prefix):
    """Check that `sub_ids` are found in `super_ids`."""
    missing_ids = _get_missing_ids(sub_ids, super_ids)

    if n_misses := len(missing_ids):
        ids_str = ", ".join(map(str, missing_ids[:10]))
        ids_str += ", ..." if n_misses > 10 else ""
        message = f"{n_misses} id(s) not found in {super_source}: {ids_str}"

        return [BluepySnapValidationError.fatal(f"{prefix}: {message}")]

    return []


def _validate_spike_file_contents(input_, config, prefix):
    try:
        spike_ids = _get_ids_from_spike_file(_resolve_path(input_["spike_file"], config))
    except IOError as e:
        return [BluepySnapValidationError.fatal(f"{prefix}: {' '.join(map(str,e.args))}")]

    nodeset_ids = _get_ids_from_node_set(input_["source"], config)
    source = f"node set '{input_['source']}'"

    return _compare_ids(spike_ids, nodeset_ids, source, prefix)


def _validate_spike_input(name, input_, config):
    errors = []

    if (key := "source") in input_:
        prefix = f"inputs.{name}.{key}"
        errors += _validate_node_set_exists(config, input_[key], prefix=prefix)

    if (key := "spike_file") in input_:
        spike_path = _resolve_path(input_[key], config)

        prefix = f"inputs.{name}.{key}"
        errors += _validate_file_exists(spike_path, prefix=prefix)

        if len(errors) > 0 or "source" not in input_ or not _file_exists(config["_circuit_config"]):
            errors += [BluepySnapValidationError.fatal(f"{prefix}: Can not validate file contents")]
        else:
            errors += _validate_spike_file_contents(input_, config, prefix)

    return errors


def _validate_input_resistance_in_nodes(_input, config, prefix):
    circuit = libsonata.CircuitConfig.from_file(config["_circuit_config"])
    node_set = _input["node_set"]
    required = "@dynamics_params/input_resistance"
    errors = []

    for pop_name, _ in _get_ids_from_node_set(node_set, config).items():
        if "input_resistance" not in circuit.node_population(pop_name).dynamics_attribute_names:
            message = f"'{required}' not found for population '{pop_name}'"
            errors += [BluepySnapValidationError.fatal(f"{prefix}: {message}")]

    return errors


def _validate_input(name, input_, config):
    """Helper function to validate a single input."""
    errors = []

    if (key := "node_set") in input_:
        prefix = f"inputs.{name}.{key}"
        errors += _validate_node_set_exists(config, input_[key], prefix)

    module = input_.get("module", "")
    if module == "synapse_replay":
        errors += _validate_spike_input(name, input_, config)
    elif "shot_noise" in module or "ornstein_uhlenbeck" in module:
        prefix = f"inputs.{name}.{module}"
        if len(errors) or "node_set" not in input_:
            message = (
                "Can not validate presence of '@dynamics_params/input_resistance' in nodes files"
            )
            errors += [BluepySnapValidationError.fatal(f"{prefix}: {message}")]
        else:
            errors += _validate_input_resistance_in_nodes(input_, config, prefix)

    return errors


def validate_inputs(config):
    """Validate the 'inputs' section in the config."""
    errors = []

    for name, input_config in config.get("inputs", {}).items():
        errors += _validate_input(name, input_config, config)

    return errors


def validate_network(config):
    """Validate the 'network' section in the config."""
    key = "network"

    if path := config.get(key, "circuit_config.json"):
        return _validate_file_exists(_resolve_path(path, config), prefix=key)

    return [BluepySnapValidationError.warning(f"{key}: circuit path not specified")]


def validate_node_set(config):
    """Validate the 'node_set' section in the config."""
    key = "node_set"

    if key in config:
        return _validate_node_set_exists(config, config[key], prefix=key)

    return []


def validate_node_sets_file(config):
    """Validate the 'node_sets_file' section in the config."""
    key = "node_sets_file"

    if key in config:
        return _validate_file_exists(config[key], prefix=key)

    return []


def validate_output(config):
    """Validate the 'output' section in the config."""
    output = config.get("output", {})
    output_dir = config["_output_dir"]

    errors = []
    if not output_dir.is_dir():
        message = f"output.output_dir: No such directory: {output_dir}"
        errors += [BluepySnapValidationError.warning(message)]

    # Only test for file existence if log file given; default is to write to STDOUT
    if (key := "log_file") in output:
        log_path = output_dir / output[key]
        errors += _validate_file_exists(log_path, prefix=f"output.{key}", fatal=False)

    spike_path = output_dir / output.get("spikes_file", "out.h5")
    errors += _validate_file_exists(spike_path, prefix="output.spikes_file", fatal=False)

    return errors


def _validate_report(name, report, config):
    """Helper function to validate a single report."""
    errors = []

    if (key := "cells") in report:
        prefix = f"reports.{name}.{key}"
        errors += _validate_node_set_exists(config, report[key], prefix)

    report_file = report.get("file_name", f"{name}.h5")
    report_file = f"{report_file}.h5" if not report_file.endswith(".h5") else report_file
    report_file = config["_output_dir"] / report_file
    errors += _validate_file_exists(report_file, prefix=f"reports.{name}.file_name", fatal=False)

    return errors


def validate_reports(config):
    """Validate the 'reports' section in the config."""
    errors = []

    for name, report in config.get("reports", {}).items():
        errors += _validate_report(name, report, config)

    return errors


def _get_ids_from_non_virtual_pops(config):
    """Get ids of all non-virtual populations."""
    circuit = libsonata.CircuitConfig.from_file(config["_circuit_config"])

    return {
        pop: circuit.node_population(pop).select_all().flatten()
        for pop in circuit.node_populations
        if circuit.node_population_properties(pop).type != "virtual"
    }


def _validate_electrodes_file(path, config):
    """Validate the ids for each of the populations in `electrodes_file` can be found."""
    prefix = "run.electrodes_file"
    errors = _validate_file_exists(path, prefix=prefix)

    if len(errors) or not _file_exists(config["_circuit_config"]):
        message = f"{prefix}: Can not validate file contents"
        return errors + [BluepySnapValidationError.fatal(message)]

    with h5py.File(path, "r") as h5:
        # root level contains population groups (per their name) and group "electrodes"
        population_names = set(h5) - {"electrodes"}
        elec_ids = {pop: np.array(h5[pop]["node_ids"]) for pop in population_names}

    if node_set := config.get("node_set"):
        source = f"node set '{node_set}'"
        sim_ids = _get_ids_from_node_set(node_set, config)
    else:
        source = "non-virtual populations"
        sim_ids = _get_ids_from_non_virtual_pops(config)

    return _compare_ids(elec_ids, sim_ids, source, prefix)


def validate_run(config):
    """Validate the 'run' section in the config."""
    if electrodes_file := config.get("run", {}).get("electrodes_file", ""):
        electrodes_file = _resolve_path(electrodes_file, config)
        return _validate_electrodes_file(electrodes_file, config)

    return []


VALIDATORS = {
    "conditions": validate_conditions,
    "connection_overrides": validate_connection_overrides,
    "inputs": validate_inputs,
    "network": validate_network,
    "node_set": validate_node_set,
    "node_sets_file": validate_node_sets_file,
    "output": validate_output,
    "reports": validate_reports,
    "run": validate_run,
}


def validate_config(config):
    """Iterate through the sections in the config and validate them."""
    return [error for section in sorted(VALIDATORS) for error in VALIDATORS[section](config)]


def validate(config_file, print_errors=True):
    """Validate Sonata simulation config.

    Args:
        config_file (str): path to Sonata simulation config file
        print_errors (bool): print errors

    Returns:
        set: set of errors, empty if no errors
    """
    config = _parse_config(config_file)
    errors = schemas.validate_simulation_schema(config_file, config)

    config = _add_validation_parameters(config, config_file)
    errors += validate_config(config)

    if print_errors:
        print_validation_errors(errors)

    return set(errors)
