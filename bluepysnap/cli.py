"""The project's command line launcher."""

import functools
import logging

import click

from bluepysnap import circuit_validation, simulation_validation

CLICK_EXISTING_FILE = click.Path(exists=True, file_okay=True, dir_okay=False)


@click.group()
@click.version_option()
@click.option("-v", "--verbose", count=True)
def cli(verbose):
    """The CLI object."""
    logging.basicConfig(
        level=(logging.WARNING, logging.INFO, logging.DEBUG)[min(verbose, 2)],
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def circuit_validation_params(func):
    """Small helper to have shared params."""

    @click.argument("config_file", type=CLICK_EXISTING_FILE)
    @click.option(
        "--skip-slow/--no-skip-slow",
        default=True,
        help=(
            "Skip slow checks; checking all edges refer to existing node ids, "
            "edge indices are correct, etc"
        ),
    )
    @click.option("--only-errors", is_flag=True, help="Only print fatal errors (ignore warnings)")
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@cli.command()
@circuit_validation_params
def validate_circuit(config_file, skip_slow, only_errors):
    """Validate Sonata circuit based on config file.

    Args:
        config_file (str): path to Sonata circuit config file
        skip_slow (bool): skip slow tests
        only_errors (bool): only print fatal errors
    """
    circuit_validation.validate(config_file, skip_slow, only_errors)


@cli.command()
@click.argument("config_file", type=CLICK_EXISTING_FILE)
def validate_simulation(config_file):
    """Validate Sonata simulation based on config file.

    Args:
        config_file (str): path to Sonata simulation config file
    """
    simulation_validation.validate(config_file)
