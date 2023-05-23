import bluepysnap as test_module


def test_version():
    result = test_module.__version__

    assert isinstance(result, str)
    assert len(result) > 0
    assert result[0].isdigit()
