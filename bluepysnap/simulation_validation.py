from pathlib import Path

from bluepysnap import schemas
from bluepysnap.config import Parser
from bluepysnap.utils import load_json, print_validation_errors


def validate(config_file, print_errors=True):
    """Validates Sonata simulation config.

    Args:
        config_file (str): path to Sonata simulation config file
        print_errors (bool): print errors

    Returns:
        set: set of errors, empty if no errors
    """
    config = Parser.parse(load_json(config_file), str(Path(config_file).parent))
    errors = schemas.validate_simulation_schema(config_file, config)

    if print_errors:
        print_validation_errors(errors)

    return set(errors)
