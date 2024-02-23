import warnings
from unittest.mock import Mock, patch

import click
import pytest
from click.testing import CliRunner

from bluepysnap.cli import cli
from bluepysnap.exceptions import BluepySnapDeprecationWarning

from utils import TEST_DATA_DIR


@patch("bluepysnap.schemas.validate_nodes_schema", Mock(return_value=[]))
@patch("bluepysnap.schemas.validate_edges_schema", Mock(return_value=[]))
@patch("bluepysnap.schemas.validate_circuit_schema", Mock(return_value=[]))
def test_cli_validate_circuit_correct():
    runner = CliRunner()
    result = runner.invoke(cli, ["validate-circuit", str(TEST_DATA_DIR / "circuit_config.json")])
    assert result.exit_code == 0
    assert click.style("No Error: Success.", fg="green") in result.stdout


def test_cli_validate_circuit_no_config():
    runner = CliRunner()
    result = runner.invoke(cli, ["validate-circuit"])
    assert result.exit_code == 2
    assert "Missing argument 'CONFIG_FILE'" in result.stdout


def test_cli_validate_simulation_correct():
    runner = CliRunner()
    result = runner.invoke(
        cli, ["validate-simulation", str(TEST_DATA_DIR / "simulation_config.json")]
    )
    assert result.exit_code == 0
    assert click.style("No Error: Success.", fg="green") in result.stdout


def test_cli_validate_simulation_no_config():
    runner = CliRunner()
    result = runner.invoke(cli, ["validate-simulation"])
    assert result.exit_code == 2
    assert "Missing argument 'CONFIG_FILE'" in result.stdout
