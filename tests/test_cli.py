from click.testing import CliRunner

from bluepysnap.cli import cli

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path

TEST_DIR = Path(__file__).resolve().parent


def test_cli_correct():
    runner = CliRunner()
    result = runner.invoke(cli, ['validate', str(TEST_DIR / 'data' / 'circuit_config.json')])
    assert result.exit_code == 0
    assert result.stdout == ''


def test_cli_no_config():
    runner = CliRunner()
    result = runner.invoke(cli, ['validate'])
    assert result.exit_code == 2
    assert 'Missing argument "CONFIG_FILE"' in result.stdout
