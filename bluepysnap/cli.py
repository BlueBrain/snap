"""The project's command line launcher."""
import logging

import click

from bluepysnap import circuit_validation


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


@cli.command()
@click.argument("config_file", type=click.Path(exists=True, file_okay=True, dir_okay=False))
def validate(config_file):
    """Validate of Sonata circuit based on config file.

    Args:
        config_file: path to Sonata circuit config file
    """
    circuit_validation.validate(config_file)
