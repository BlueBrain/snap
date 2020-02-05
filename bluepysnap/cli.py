"""The project's command line launcher."""
import click

from bluepysnap.circuit_validation import validate


@click.group()
def cli():
    """The CLI object."""


@cli.command('validate', short_help='Validate Sonata circuit')
@click.argument('config_file', type=click.Path(exists=True, file_okay=True, dir_okay=False))
def validate_cli(config_file):
    """Cli command for validating of Sonata circuit.

    Args:
        config_file: path to Sonata circuit config file
    """
    validate(config_file)
