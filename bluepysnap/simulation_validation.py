"""Standalone module that validates Sonata simulation. See ``validate-simulation`` function."""
import contextlib
import io
import os
from pathlib import Path

from bluepysnap import schemas
from bluepysnap.config import Parser
from bluepysnap.exceptions import BluepySnapValidationError
from bluepysnap.node_sets import NodeSets
from bluepysnap.utils import load_json, print_validation_errors

try:
    NEURODAMUS_PRESENT = True
    import neurodamus
except ImportError:
    NEURODAMUS_PRESENT = False


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
    output_dir = Path(config.get("output", {}).get("output_dir", "output"))
    return output_dir if output_dir.is_absolute() else config["_config_dir"] / output_dir


def _add_validation_parameters(config, config_path):
    """Add helper parameters to the config."""
    config["_config_dir"] = Path(config_path).parent.absolute()
    config["_output_dir"] = _get_output_dir(config)
    config["_node_sets_instance"] = _get_node_sets(config)

    return config


@contextlib.contextmanager
def _silent_neurodamus():
    """Yield a no-log, no-output (unless error) NeurodamusCore instance."""
    # No need to init MPI since we're not running anything. This also supresses some output.
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


def _validate_file_exists(path, fatal=True, prefix=None):
    """Validates the existence of a file.

    Note: Error is the same if ``path`` is a directory.
    """
    error = BluepySnapValidationError.fatal if fatal else BluepySnapValidationError.warning
    msg = f"{prefix}: " if prefix else ""
    return [] if _file_exists(path) else [error(f"{msg}No such file: {path}")]


def _validate_node_set_exists(config, node_set, prefix=None):
    """Validates the existence of a node set."""
    node_sets = config["_node_sets_instance"]
    msg = f"{prefix}: " if prefix else ""
    if node_set not in node_sets:
        return [BluepySnapValidationError.fatal(f"{msg}Unknown node set: '{node_set}'")]

    return []


def _validate_mechanism_variables(name, mechanism, prefix):
    """Check that mechanism name (=suffix) and corresponding variables are defined in neurodamus."""
    nd_prefix = f"{prefix}.{name}: neurodamus"

    with _silent_neurodamus() as nd:
        nd_attrs = dir(nd.h)

    if name not in nd_attrs:
        return [BluepySnapValidationError.fatal(f"{nd_prefix}: Unknown SUFFIX: {name}")]

    errors = []
    for variable in mechanism:
        if (suffix_name := f"{variable}_{name}") not in nd_attrs:
            message = f"{nd_prefix}: Unknown variable: {suffix_name}"
            errors += [BluepySnapValidationError.fatal(message)]

    return errors


def _validate_mechanisms(mechanisms, prefix):
    """Validate the 'conditions.mechanisms' section in the config."""
    if NEURODAMUS_PRESENT:
        errors = []
        for name, mechanism in mechanisms.items():
            errors += _validate_mechanism_variables(name, mechanism, prefix)

        return errors

    return _warn_on_no_neurodamus(prefix)


def validate_conditions(config):
    """Validate the 'conditions' section in the config."""
    key = "conditions"
    node_set_key = "node_set"
    mech_key = "mechanisms"
    mod_key = "modifications"
    conditions = config.get(key, {})
    errors = []

    for mod_name, mod_config in conditions.get(mod_key, {}).items():
        if node_set_key in mod_config:
            errors += _validate_node_set_exists(
                config,
                mod_config[node_set_key],
                prefix=f"{key}.{mod_key}.{mod_name}.{node_set_key}",
            )

    if mech_key in conditions:
        errors += _validate_mechanisms(conditions[mech_key], prefix=f"{key}.{mech_key}")

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


def _validate_input(name, input_, config):
    """Helper function to validate a single input."""
    key = "inputs"
    node_set_key = "node_set"
    errors = []

    if (node_set := input_.get(node_set_key)) is not None:
        errors += _validate_node_set_exists(config, node_set, prefix=f"{key}.{name}.{node_set_key}")

    if input_.get("module") == "synapse_replay":
        spike_key = "spike_file"
        if (spike_path := input_.get(spike_key)) is not None:
            if not Path(spike_path).is_absolute():
                spike_path = config["_config_dir"] / spike_path
            errors += _validate_file_exists(spike_path, prefix=f"{key}.{name}.{spike_key}")

        node_set_key = "source"
        if (node_set := input_.get(node_set_key)) is not None:
            errors += _validate_node_set_exists(
                config, node_set, prefix=f"{key}.{name}.{node_set_key}"
            )

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
    if key in config:
        return _validate_file_exists(config[key], prefix=key)
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
    key = "output"
    output = config.get(key, {})
    output_dir = config["_output_dir"]

    if output_dir.is_dir():
        # TODO: Should the warnings be also added when the output folder is missing? Likely, yes.
        errors = []

        prop = "log_file"
        if prop in output:
            # Only test for file existence if log file given; default is to write to STDOUT
            log_path = output_dir / output[prop]
            errors += _validate_file_exists(log_path, fatal=False, prefix=f"{key}.{prop}")

        prop = "spikes_file"
        spike_path = output_dir / output.get(prop, "out.h5")
        errors += _validate_file_exists(spike_path, fatal=False, prefix=f"{key}.{prop}")

        return errors

    return [BluepySnapValidationError.warning(f"{key}.output_dir: No such directory: {output_dir}")]


def _validate_report(name, report, config):
    """Helper function to validate a single report."""
    key = "reports"
    node_set_key = "cells"
    errors = []

    if (node_set := report.get(node_set_key)) is not None:
        errors += _validate_node_set_exists(config, node_set, prefix=f"{key}.{name}.{node_set_key}")

    file_key = "file_name"
    report_file = report.get(file_key, f"{name}.h5")
    report_file = f"{report_file}.h5" if not report_file.endswith(".h5") else report_file
    report_file = config["_output_dir"] / report_file
    errors += _validate_file_exists(report_file, fatal=False, prefix=f"{key}.{name}.{file_key}")

    return errors


def validate_reports(config):
    """Validate the 'reports' section in the config."""
    errors = []

    for name, report in config.get("reports", {}).items():
        errors += _validate_report(name, report, config)

    return errors


def validate_run(config):
    """Validate the 'run' section in the config."""
    key = "run"
    prop = "electrodes_file"
    if key in config and prop in config[key]:
        return _validate_file_exists(config[key][prop], prefix=f"{key}.{prop}")

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
