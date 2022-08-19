import click
from click.testing import CliRunner
from mock import Mock, patch

from bluepysnap.cli import cli

from utils import TEST_DATA_DIR


@patch("bluepysnap.schemas.validate_nodes_schema", Mock(return_value=[]))
@patch("bluepysnap.schemas.validate_edges_schema", Mock(return_value=[]))
@patch("bluepysnap.schemas.validate_circuit_schema", Mock(return_value=[]))
def test_cli_correct():
    runner = CliRunner()
    result = runner.invoke(cli, ["validate", str(TEST_DATA_DIR / "circuit_config.json")])
    assert result.exit_code == 0
    assert click.style("No Error: Success.", fg="green") in result.stdout


def test_cli_no_config():
    runner = CliRunner()
    result = runner.invoke(cli, ["validate"])
    assert result.exit_code == 2
    assert "Missing argument 'CONFIG_FILE'" in result.stdout
